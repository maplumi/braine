use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

pub(super) fn download_bytes(filename: &str, bytes: &[u8]) -> Result<(), String> {
    let window = web_sys::window().ok_or("no window".to_string())?;
    let document = window.document().ok_or("no document".to_string())?;

    let array = js_sys::Uint8Array::from(bytes);
    let parts = js_sys::Array::new();
    parts.push(&array.buffer());
    let blob = web_sys::Blob::new_with_u8_array_sequence(&parts)
        .map_err(|_| "blob: failed to create".to_string())?;

    let url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|_| "url: create_object_url failed".to_string())?;

    let a = document
        .create_element("a")
        .map_err(|_| "document: create_element failed".to_string())?
        .dyn_into::<web_sys::HtmlAnchorElement>()
        .map_err(|_| "document: anchor cast failed".to_string())?;

    a.set_href(&url);
    a.set_download(filename);
    a.click();

    let _ = web_sys::Url::revoke_object_url(&url);
    Ok(())
}

pub(super) async fn read_file_bytes(file: web_sys::File) -> Result<Vec<u8>, String> {
    let promise = file_reader_array_buffer_promise(file)?;
    let v = wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .map_err(|_| "file: read failed".to_string())?;

    let buf = v
        .dyn_into::<js_sys::ArrayBuffer>()
        .map_err(|_| "file: expected ArrayBuffer".to_string())?;
    let arr = js_sys::Uint8Array::new(&buf);
    let mut out = vec![0u8; arr.length() as usize];
    arr.copy_to(&mut out);
    Ok(out)
}

fn file_reader_array_buffer_promise(file: web_sys::File) -> Result<js_sys::Promise, String> {
    let reader =
        web_sys::FileReader::new().map_err(|_| "file: FileReader::new failed".to_string())?;
    reader
        .read_as_array_buffer(&file)
        .map_err(|_| "file: read_as_array_buffer failed".to_string())?;

    Ok(js_sys::Promise::new(&mut |resolve, reject| {
        let reject_load = reject.clone();
        let reject_err = reject;
        let reader_ok = reader.clone();
        let onload =
            Closure::wrap(Box::new(
                move |_ev: web_sys::ProgressEvent| match reader_ok.result() {
                    Ok(v) => {
                        if v.is_null() || v.is_undefined() {
                            let _ = reject_load.call1(
                                &JsValue::UNDEFINED,
                                &JsValue::from_str("file: missing result"),
                            );
                        } else {
                            let _ = resolve.call1(&JsValue::UNDEFINED, &v);
                        }
                    }
                    Err(_) => {
                        let _ = reject_load.call1(
                            &JsValue::UNDEFINED,
                            &JsValue::from_str("file: result() threw"),
                        );
                    }
                },
            ) as Box<dyn FnMut(_)>);
        reader.set_onload(Some(onload.as_ref().unchecked_ref()));
        onload.forget();

        let onerror = Closure::wrap(Box::new(move |_ev: web_sys::ProgressEvent| {
            let _ = reject_err.call1(&JsValue::UNDEFINED, &JsValue::from_str("file: read error"));
        }) as Box<dyn FnMut(_)>);
        reader.set_onerror(Some(onerror.as_ref().unchecked_ref()));
        onerror.forget();
    }))
}
