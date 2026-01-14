pub(super) fn choose_text_token_sensor(last_byte: Option<u8>, known_sensors: &[String]) -> String {
    let preferred = match last_byte {
        Some(b) => format!("txt_tok_{b:02X}"),
        None => "txt_tok_UNK".to_string(),
    };

    if known_sensors.iter().any(|s| s == &preferred) {
        return preferred;
    }

    let unk = "txt_tok_UNK";
    if known_sensors.iter().any(|s| s == unk) {
        return unk.to_string();
    }

    known_sensors.first().cloned().unwrap_or(preferred)
}

pub(super) fn token_action_name_from_sensor(sensor: &str) -> String {
    match sensor.strip_prefix("txt_tok_") {
        Some(suffix) => format!("tok_{suffix}"),
        None => "tok_UNK".to_string(),
    }
}

pub(super) fn display_token_from_action(action: &str) -> String {
    let Some(suffix) = action.strip_prefix("tok_") else {
        return "<unk>".to_string();
    };

    if suffix == "UNK" {
        return "<unk>".to_string();
    }

    if suffix.len() != 2 {
        return format!("<{suffix}>");
    }

    let Ok(b) = u8::from_str_radix(suffix, 16) else {
        return format!("<{suffix}>");
    };

    if b == b' ' {
        "<sp>".to_string()
    } else if b == b'\n' {
        "\\n".to_string()
    } else if (0x21..=0x7E).contains(&b) {
        (b as char).to_string()
    } else {
        format!("0x{b:02X}")
    }
}
