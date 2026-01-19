use std::io::{self, Read, Write};

pub const MAGIC: &[u8; 8] = b"BRAINE01";
pub const VERSION_V3: u32 = 3;
pub const VERSION_CURRENT: u32 = VERSION_V3;

pub fn compress_lz4(input: &[u8]) -> Vec<u8> {
    lz4_flex::compress(input)
}

pub fn decompress_lz4(input: &[u8], expected_size: usize) -> io::Result<Vec<u8>> {
    // Strict format: raw LZ4 block with external expected size.
    lz4_flex::decompress(input, expected_size)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "lz4 decompression failed"))
}

pub struct CapacityWriter<W> {
    inner: W,
    remaining: usize,
    written: usize,
}

pub struct CountingWriter {
    written: usize,
}

impl CountingWriter {
    pub fn new() -> Self {
        Self { written: 0 }
    }

    pub fn written(&self) -> usize {
        self.written
    }
}

impl Default for CountingWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl Write for CountingWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.written = self.written.saturating_add(buf.len());
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<W: Write> CapacityWriter<W> {
    pub fn new(inner: W, capacity_bytes: usize) -> Self {
        Self {
            inner,
            remaining: capacity_bytes,
            written: 0,
        }
    }

    pub fn remaining(&self) -> usize {
        self.remaining
    }

    pub fn written(&self) -> usize {
        self.written
    }

    pub fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: Write> Write for CapacityWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.len() > self.remaining {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "CapacityWriter: out of space",
            ));
        }
        let n = self.inner.write(buf)?;
        self.remaining = self.remaining.saturating_sub(n);
        self.written = self.written.saturating_add(n);
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

pub fn write_u32_le<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub fn write_u64_le<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub fn write_f32_le<W: Write>(w: &mut W, v: f32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub fn write_bytes<W: Write>(w: &mut W, bytes: &[u8]) -> io::Result<()> {
    write_u32_le(w, bytes.len() as u32)?;
    w.write_all(bytes)
}

pub fn write_string<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    write_bytes(w, s.as_bytes())
}

pub fn read_exact<const N: usize, R: Read>(r: &mut R) -> io::Result<[u8; N]> {
    let mut buf = [0u8; N];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

pub fn read_u32_le<R: Read>(r: &mut R) -> io::Result<u32> {
    Ok(u32::from_le_bytes(read_exact::<4, _>(r)?))
}

pub fn read_u64_le<R: Read>(r: &mut R) -> io::Result<u64> {
    Ok(u64::from_le_bytes(read_exact::<8, _>(r)?))
}

pub fn read_f32_le<R: Read>(r: &mut R) -> io::Result<f32> {
    Ok(f32::from_le_bytes(read_exact::<4, _>(r)?))
}

pub fn read_bytes<R: Read>(r: &mut R) -> io::Result<Vec<u8>> {
    let n = read_u32_le(r)? as usize;
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

pub fn read_string<R: Read>(r: &mut R) -> io::Result<String> {
    let bytes = read_bytes(r)?;
    String::from_utf8(bytes)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid utf-8 string"))
}

pub fn write_chunk<W: Write>(w: &mut W, tag: [u8; 4], payload: &[u8]) -> io::Result<()> {
    w.write_all(&tag)?;
    write_u32_le(w, payload.len() as u32)?;
    w.write_all(payload)
}

/// Write a V2 chunk: payload is LZ4-compressed and preceded by the uncompressed length (u32).
///
/// Layout:
/// - tag: [u8;4]
/// - len: u32 (bytes following, including the 4-byte uncompressed length)
/// - uncompressed_len: u32
/// - compressed payload bytes
pub fn write_chunk_v2_lz4<W: Write>(w: &mut W, tag: [u8; 4], payload: &[u8]) -> io::Result<()> {
    let compressed = compress_lz4(payload);
    let uncompressed_len = payload.len() as u32;
    let total_len = 4u32.saturating_add(
        u32::try_from(compressed.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "chunk too large"))?,
    );

    w.write_all(&tag)?;
    write_u32_le(w, total_len)?;
    write_u32_le(w, uncompressed_len)?;
    w.write_all(&compressed)
}

pub fn read_chunk_header<R: Read>(r: &mut R) -> io::Result<([u8; 4], u32)> {
    let tag = read_exact::<4, _>(r)?;
    let len = read_u32_le(r)?;
    Ok((tag, len))
}
