use lz4_flex::decompress_into;
use memmap::{Mmap, MmapOptions};
use std::{fs, path};
use wkwrap::{BlockType, Header, Result, Vec3};

fn morton_encode(vec: &Vec3) -> u64 {
  let x = vec.x as u64;
  let y = vec.y as u64;
  let z = vec.z as u64;
  let mut morton = 0u64;
  let bit_length = 64 - (std::cmp::max(x, std::cmp::max(y, z)) + 1).leading_zeros();

  for i in 0..bit_length {
    morton |= ((x & (1 << i)) << (2 * i))
      | ((y & (1 << i)) << (2 * i + 1))
      | ((z & (1 << i)) << (2 * i + 2))
  }
  morton
}

#[derive(Debug)]
pub struct File {
  file_mmap: Mmap,
  header: Header,
}

impl File {
  fn new(file: fs::File, header: Header) -> Result<File> {
    Ok(File {
      file_mmap: unsafe {
        MmapOptions::new().map(&file).or(Err(String::from(
          "Could not open WKW file as memory-mapped file",
        )))?
      },
      header,
    })
  }

  pub fn open(path: &path::Path) -> Result<File> {
    let mut file = fs::File::open(path).or(Err(format!("Could not open WKW file {:?}", path)))?;
    let header = Header::read(&mut file)?;
    Ok(Self::new(file, header)?)
  }

  pub fn read_block(&self, src_pos: Vec3, buf: &mut [u8]) -> Result<()> {
    let file_len_vx = self.header.file_len_vx();
    let block_len_log2 = self.header.block_len_log2 as u32;
    let file_len_vx_vec = Vec3::from(file_len_vx);
    assert!(src_pos < file_len_vx_vec);

    let block_id = src_pos >> block_len_log2;
    let block_idx = morton_encode(&block_id);

    if buf.len() != self.header.block_size() {
      return Err(String::from("Buffer has invalid size"));
    }

    match self.header.block_type {
      BlockType::Raw => self.read_block_raw(block_idx, buf)?,
      BlockType::LZ4 | BlockType::LZ4HC => self.read_block_lz4(block_idx, buf)?,
    };

    Ok(())
  }

  fn read_block_raw(&self, block_idx: u64, buf: &mut [u8]) -> Result<usize> {
    let block_offset = self.header.block_offset(block_idx)? as usize;
    let block_size = self.header.block_size();

    let buf_from_disk = &self
      .file_mmap
      .get(block_offset..(block_offset + block_size))
      .ok_or(String::from("Could not read block from disk"))?;

    buf.copy_from_slice(buf_from_disk);
    Ok(block_size)
  }

  fn read_block_lz4(&self, block_idx: u64, buf: &mut [u8]) -> Result<usize> {
    let block_offset = self.header.block_offset(block_idx)? as usize;
    let block_size_lz4 = self.header.block_size_on_disk(block_idx)?;
    let block_size_raw = self.header.block_size();

    // read compressed block
    let buf_from_disk = &self
      .file_mmap
      .get(block_offset..(block_offset + block_size_lz4))
      .ok_or(String::from("Could not read block from disk"))?;

    // decompress block
    let byte_written = match decompress_into(&buf_from_disk, buf, 0) {
      Ok(byte_written) => byte_written,
      Err(_) => return Err(String::from("Error in LZ4 decompress")),
    };

    match byte_written == block_size_raw {
      true => Ok(byte_written),
      false => Err(String::from("Unexpected length after decompression")),
    }
  }
}
