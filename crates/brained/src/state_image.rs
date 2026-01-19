use braine::{storage, substrate::Brain};
use std::io::{self, Read, Write};

pub const MAGIC_V3: &[u8; 8] = b"BRSTATE3";

pub const VERSION_V3: u32 = 3;

/// Tags for chunks in the persisted daemon state file.
const TAG_BRAIN_IMAGE: [u8; 4] = *b"BIMG";
const TAG_EXPERTS_STATE: [u8; 4] = *b"EXPT";
const TAG_RUNTIME_STATE: [u8; 4] = *b"RTST";

pub struct LoadedState {
    pub brain: Brain,
    pub experts_state: Option<Vec<u8>>,
    pub runtime_state: Option<Vec<u8>>,
}

pub fn is_state_magic(magic: &[u8; 8]) -> bool {
    magic == MAGIC_V3
}

pub fn save_state_to_with_version<W: Write>(
    w: &mut W,
    brain: &Brain,
    experts_state: &[u8],
    runtime_state: Option<&[u8]>,
    version: u32,
) -> io::Result<()> {
    if version != VERSION_V3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "unsupported state version",
        ));
    }

    w.write_all(MAGIC_V3)?;
    storage::write_u32_le(w, VERSION_V3)?;

    let mut brain_bytes: Vec<u8> = Vec::new();
    // State wrapper stores the current compressed brain image bytes (which are already chunked).
    brain.save_image_to(&mut brain_bytes)?;
    storage::write_chunk_v2_lz4(w, TAG_BRAIN_IMAGE, &brain_bytes)?;
    storage::write_chunk_v2_lz4(w, TAG_EXPERTS_STATE, experts_state)?;
    if let Some(rt) = runtime_state {
        storage::write_chunk_v2_lz4(w, TAG_RUNTIME_STATE, rt)?;
    }
    Ok(())
}

pub fn load_state_from<R: Read>(r: &mut R) -> io::Result<LoadedState> {
    let magic = storage::read_exact::<8, _>(r)?;
    if !is_state_magic(&magic) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bad state magic",
        ));
    }

    let version = storage::read_u32_le(r)?;
    if version != VERSION_V3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported state version",
        ));
    }

    let mut brain_img: Option<Vec<u8>> = None;
    let mut experts_state: Option<Vec<u8>> = None;
    let mut runtime_state: Option<Vec<u8>> = None;

    loop {
        let (tag, len) = match storage::read_chunk_header(r) {
            Ok(v) => v,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        };

        let buf = {
            let mut take = r.take(len as u64);
            let uncompressed_len = storage::read_u32_le(&mut take)? as usize;
            let mut compressed = Vec::with_capacity((len as usize).saturating_sub(4));
            take.read_to_end(&mut compressed)?;
            let decompressed = storage::decompress_lz4(&compressed, uncompressed_len)?;
            io::copy(&mut take, &mut io::sink())?;
            decompressed
        };

        if tag == TAG_BRAIN_IMAGE {
            brain_img = Some(buf);
        } else if tag == TAG_EXPERTS_STATE {
            experts_state = Some(buf);
        } else if tag == TAG_RUNTIME_STATE {
            runtime_state = Some(buf);
        }
    }

    let brain_img = brain_img
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing brain image chunk"))?;
    let mut cursor = std::io::Cursor::new(brain_img);
    let brain = Brain::load_image_from(&mut cursor)?;

    Ok(LoadedState {
        brain,
        experts_state,
        runtime_state,
    })
}
