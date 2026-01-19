use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

pub(super) async fn idb_put_bytes(key: &str, bytes: &[u8]) -> Result<(), String> {
    let db = idb_open().await?;
    let tx = db
        .transaction_with_str_and_mode(super::IDB_STORE, web_sys::IdbTransactionMode::Readwrite)
        .map_err(|_| "indexeddb: failed to open transaction".to_string())?;
    let store = tx
        .object_store(super::IDB_STORE)
        .map_err(|_| "indexeddb: failed to open object store".to_string())?;

    let value = js_sys::Uint8Array::from(bytes).into();
    let req = store
        .put_with_key(&value, &JsValue::from_str(key))
        .map_err(|_| "indexeddb: put() threw".to_string())?;
    idb_request_done(req).await?;
    Ok(())
}

pub(super) async fn idb_get_bytes(key: &str) -> Result<Option<Vec<u8>>, String> {
    let db = idb_open().await?;
    let tx = db
        .transaction_with_str_and_mode(super::IDB_STORE, web_sys::IdbTransactionMode::Readonly)
        .map_err(|_| "indexeddb: failed to open transaction".to_string())?;
    let store = tx
        .object_store(super::IDB_STORE)
        .map_err(|_| "indexeddb: failed to open object store".to_string())?;

    let req = store
        .get(&JsValue::from_str(key))
        .map_err(|_| "indexeddb: get() threw".to_string())?;
    let v = idb_request_result(req).await?;
    if v.is_undefined() || v.is_null() {
        return Ok(None);
    }

    let arr = js_sys::Uint8Array::new(&v);
    let mut out = vec![0u8; arr.length() as usize];
    arr.copy_to(&mut out);
    Ok(Some(out))
}

pub(super) async fn idb_delete_key(key: &str) -> Result<(), String> {
    let db = idb_open().await?;
    let tx = db
        .transaction_with_str_and_mode(super::IDB_STORE, web_sys::IdbTransactionMode::Readwrite)
        .map_err(|_| "indexeddb: failed to open transaction".to_string())?;
    let store = tx
        .object_store(super::IDB_STORE)
        .map_err(|_| "indexeddb: failed to open object store".to_string())?;

    let req = store
        .delete(&JsValue::from_str(key))
        .map_err(|_| "indexeddb: delete() threw".to_string())?;
    idb_request_done(req).await?;
    Ok(())
}

/// Save game accuracies to IndexedDB as JSON
pub(super) async fn save_game_accuracies(
    accs: &std::collections::HashMap<String, f32>,
) -> Result<(), String> {
    let json = serde_json::to_string(accs).map_err(|e| format!("serialize error: {}", e))?;
    idb_put_bytes(super::IDB_KEY_GAME_ACCURACY, json.as_bytes()).await
}

/// Load game accuracies from IndexedDB
pub(super) async fn load_game_accuracies() -> Result<std::collections::HashMap<String, f32>, String>
{
    match idb_get_bytes(super::IDB_KEY_GAME_ACCURACY).await? {
        Some(bytes) => {
            let json = String::from_utf8(bytes).map_err(|_| "invalid utf8")?;
            serde_json::from_str(&json).map_err(|e| format!("parse error: {}", e))
        }
        None => Ok(std::collections::HashMap::new()),
    }
}

async fn idb_open() -> Result<web_sys::IdbDatabase, String> {
    let promise = idb_open_promise()?;
    let v = wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .map_err(|_| "indexeddb: open() failed".to_string())?;
    v.dyn_into::<web_sys::IdbDatabase>()
        .map_err(|_| "indexeddb: open() returned unexpected type".to_string())
}

fn idb_open_promise() -> Result<js_sys::Promise, String> {
    let w = web_sys::window().ok_or("no window")?;
    let factory = w
        .indexed_db()
        .map_err(|_| "indexeddb() threw".to_string())?
        .ok_or("indexeddb unavailable".to_string())?;

    let req = factory
        .open_with_u32(super::IDB_DB_NAME, 1)
        .map_err(|_| "indexeddb: open_with_u32() threw".to_string())?;

    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        let resolve = resolve.clone();
        let reject_upgrade = reject.clone();
        let reject_success = reject.clone();
        let reject_error = reject;

        // Upgrade: create the object store.
        let on_upgrade = Closure::wrap(Box::new(move |ev: web_sys::Event| {
            let Some(target) = ev.target() else {
                let _ = reject_upgrade.call1(
                    &JsValue::UNDEFINED,
                    &JsValue::from_str("indexeddb: no event target"),
                );
                return;
            };
            let Ok(open_req) = target.dyn_into::<web_sys::IdbOpenDbRequest>() else {
                let _ = reject_upgrade.call1(
                    &JsValue::UNDEFINED,
                    &JsValue::from_str("indexeddb: bad upgrade target"),
                );
                return;
            };
            let db = match open_req.result() {
                Ok(v) => match v.dyn_into::<web_sys::IdbDatabase>() {
                    Ok(db) => db,
                    Err(_) => {
                        let _ = reject_upgrade.call1(
                            &JsValue::UNDEFINED,
                            &JsValue::from_str("indexeddb: upgrade result not a db"),
                        );
                        return;
                    }
                },
                Err(_) => {
                    let _ = reject_upgrade.call1(
                        &JsValue::UNDEFINED,
                        &JsValue::from_str("indexeddb: upgrade result() threw"),
                    );
                    return;
                }
            };

            // Creating an existing store throws; ignore if it already exists.
            let _ = db.create_object_store(super::IDB_STORE);
        }) as Box<dyn FnMut(_)>);
        req.set_onupgradeneeded(Some(on_upgrade.as_ref().unchecked_ref()));
        on_upgrade.forget();

        let on_success = Closure::wrap(Box::new(move |ev: web_sys::Event| {
            let Some(target) = ev.target() else {
                let _ = reject_success.call1(
                    &JsValue::UNDEFINED,
                    &JsValue::from_str("indexeddb: no event target"),
                );
                return;
            };
            let Ok(open_req) = target.dyn_into::<web_sys::IdbOpenDbRequest>() else {
                let _ = reject_success.call1(
                    &JsValue::UNDEFINED,
                    &JsValue::from_str("indexeddb: bad success target"),
                );
                return;
            };
            let db = match open_req.result() {
                Ok(v) => v,
                Err(_) => {
                    let _ = reject_success.call1(
                        &JsValue::UNDEFINED,
                        &JsValue::from_str("indexeddb: result() threw"),
                    );
                    return;
                }
            };
            let _ = resolve.call1(&JsValue::UNDEFINED, &db);
        }) as Box<dyn FnMut(_)>);
        req.set_onsuccess(Some(on_success.as_ref().unchecked_ref()));
        on_success.forget();

        let on_error = Closure::wrap(Box::new(move |_ev: web_sys::Event| {
            let _ = reject_error.call1(
                &JsValue::UNDEFINED,
                &JsValue::from_str("indexeddb: open error"),
            );
        }) as Box<dyn FnMut(_)>);
        req.set_onerror(Some(on_error.as_ref().unchecked_ref()));
        on_error.forget();
    });

    Ok(promise)
}

async fn idb_request_done(req: web_sys::IdbRequest) -> Result<(), String> {
    let promise = idb_request_done_promise(req);
    wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .map(|_| ())
        .map_err(|_| "indexeddb: request failed".to_string())
}

async fn idb_request_result(req: web_sys::IdbRequest) -> Result<JsValue, String> {
    let promise = idb_request_result_promise(req);
    wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .map_err(|_| "indexeddb: request failed".to_string())
}

fn idb_request_done_promise(req: web_sys::IdbRequest) -> js_sys::Promise {
    js_sys::Promise::new(&mut |resolve, reject| {
        let on_success = Closure::wrap(Box::new(move |_ev: web_sys::Event| {
            let _ = resolve.call0(&JsValue::UNDEFINED);
        }) as Box<dyn FnMut(_)>);
        req.set_onsuccess(Some(on_success.as_ref().unchecked_ref()));
        on_success.forget();

        let on_error = Closure::wrap(Box::new(move |_ev: web_sys::Event| {
            let _ = reject.call1(
                &JsValue::UNDEFINED,
                &JsValue::from_str("indexeddb: request error"),
            );
        }) as Box<dyn FnMut(_)>);
        req.set_onerror(Some(on_error.as_ref().unchecked_ref()));
        on_error.forget();
    })
}

fn idb_request_result_promise(req: web_sys::IdbRequest) -> js_sys::Promise {
    js_sys::Promise::new(&mut |resolve, reject| {
        let reject_success = reject.clone();
        let reject_error = reject;
        let on_success = Closure::wrap(Box::new(move |ev: web_sys::Event| {
            let Some(target) = ev.target() else {
                let _ = reject_success.call1(
                    &JsValue::UNDEFINED,
                    &JsValue::from_str("indexeddb: no event target"),
                );
                return;
            };
            let Ok(req) = target.dyn_into::<web_sys::IdbRequest>() else {
                let _ = reject_success.call1(
                    &JsValue::UNDEFINED,
                    &JsValue::from_str("indexeddb: bad request target"),
                );
                return;
            };
            let v = match req.result() {
                Ok(v) => v,
                Err(_) => {
                    let _ = reject_success.call1(
                        &JsValue::UNDEFINED,
                        &JsValue::from_str("indexeddb: result() threw"),
                    );
                    return;
                }
            };
            let _ = resolve.call1(&JsValue::UNDEFINED, &v);
        }) as Box<dyn FnMut(_)>);
        req.set_onsuccess(Some(on_success.as_ref().unchecked_ref()));
        on_success.forget();

        let on_error = Closure::wrap(Box::new(move |_ev: web_sys::Event| {
            let _ = reject_error.call1(
                &JsValue::UNDEFINED,
                &JsValue::from_str("indexeddb: request error"),
            );
        }) as Box<dyn FnMut(_)>);
        req.set_onerror(Some(on_error.as_ref().unchecked_ref()));
        on_error.forget();
    })
}
