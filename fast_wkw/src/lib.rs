use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::Path;

mod wkw_dataset;
mod wkw_file;

use crate::wkw_dataset::CachedDataset;

fn convert_dtype(voxel_type: &wkwrap::VoxelType) -> String {
  String::from(match voxel_type {
    wkwrap::VoxelType::U8 => "uint8",
    wkwrap::VoxelType::U16 => "uint16",
    wkwrap::VoxelType::U32 => "uint32",
    wkwrap::VoxelType::U64 => "uint64",
    wkwrap::VoxelType::F32 => "float32",
    wkwrap::VoxelType::F64 => "float64",
    wkwrap::VoxelType::I8 => "int8",
    wkwrap::VoxelType::I16 => "int16",
    wkwrap::VoxelType::I32 => "int32",
    wkwrap::VoxelType::I64 => "int64",
  })
}

#[pyclass]
struct Block {
  #[pyo3(get)]
  buf: PyObject,
  #[pyo3(get)]
  dtype: String,
  #[pyo3(get)]
  shape: (usize, usize, usize, usize),
}

#[pyclass]
struct DatasetRepository {
  file_cache: wkw_dataset::FileCache,
}

#[pymethods]
impl DatasetRepository {
  #[new]
  fn new(cap: usize) -> Self {
    DatasetRepository {
      file_cache: wkw_dataset::FileCache::new(cap),
    }
  }

  fn get_dataset(&self, path: String) -> DatasetHandle {
    DatasetHandle {
      dataset: CachedDataset::new(Path::new(&path), self.file_cache.clone()).unwrap(),
    }
  }

  fn clear_cache_prefix(&self, path_prefix: String) -> PyResult<()> {
    self.file_cache.clear_prefix(Path::new(&path_prefix));
    Ok(())
  }
}

#[pyclass]
struct DatasetHandle {
  dataset: CachedDataset,
}

#[pymethods]
impl DatasetHandle {
  fn read_block(&self, py: Python, src_pos: (u32, u32, u32)) -> PyResult<PyObject> {
    let dataset = self.dataset.clone();
    pyo3_asyncio::tokio::into_coroutine(py, async move {
      let (buf, dtype, shape) = tokio::task::spawn_blocking(move || {
        let offset = wkwrap::Vec3 {
          x: src_pos.0,
          y: src_pos.1,
          z: src_pos.2,
        };
        let mut buf = vec![0; dataset.header().block_size()];
        dataset.read_block(offset, &mut buf).unwrap();
        (
          buf,
          convert_dtype(&dataset.header().voxel_type),
          (
            dataset.header().num_channels(),
            dataset.header().block_len() as usize,
            dataset.header().block_len() as usize,
            dataset.header().block_len() as usize,
          ),
        )
      })
      .await
      .unwrap();
      Python::with_gil(|py| {
        let py_bytes = PyBytes::new(py, &buf);
        let py_bytes: PyObject = py_bytes.into();
        Ok(
          Py::new(
            py,
            Block {
              buf: py_bytes,
              dtype,
              shape,
            },
          )
          .unwrap()
          .into_py(py),
        )
      })
    })
  }
}

#[pymodule]
fn fast_wkw(py: Python, m: &PyModule) -> PyResult<()> {
  pyo3_asyncio::try_init(py)?;
  pyo3_asyncio::tokio::init_multi_thread_once();

  m.add_class::<DatasetRepository>()?;
  m.add_class::<DatasetHandle>()?;

  Ok(())
}
