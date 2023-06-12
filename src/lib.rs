//! Capture the screen with DXGI Desktop Duplication

#![cfg(windows)]

pub mod pixfmt;
pub use pixfmt::{BGRA8, RGBA8};
use winapi::ctypes::c_void;
use winapi::shared::minwindef::UINT;

extern crate winapi;
extern crate wio;

use std::mem::zeroed;
use std::{mem, ptr, slice};

use rayon::prelude::*;

use winapi::shared::dxgi::{
    CreateDXGIFactory1, IDXGIAdapter, IDXGIAdapter1, IDXGIFactory1, IDXGIOutput, IDXGISurface1,
    IID_IDXGIFactory1, DXGI_MAP_READ, DXGI_OUTPUT_DESC, DXGI_RESOURCE_PRIORITY_MAXIMUM,
};
use winapi::shared::dxgi1_2::{
    IDXGIOutput1, IDXGIOutputDuplication, DXGI_OUTDUPL_POINTER_SHAPE_INFO,
};
use winapi::shared::dxgitype::*;
// use winapi::shared::ntdef::*;
use winapi::shared::windef::*;
use winapi::shared::winerror::*;
use winapi::um::d3d11::*;
use winapi::um::d3dcommon::*;
use winapi::um::unknwnbase::*;
use winapi::um::winuser::*;
use wio::com::ComPtr;

/// Possible errors when capturing
#[derive(Debug)]
pub enum CaptureError {
    /// Could not duplicate output, access denied. Might be in protected fullscreen.
    AccessDenied,
    /// Access to the duplicated output was lost. Likely, mode was changed e.g. window => full
    AccessLost,
    /// Error when trying to refresh outputs after some failure.
    RefreshFailure,
    /// AcquireNextFrame timed out.
    Timeout,
    /// General/Unexpected failure
    Fail(&'static str),
}

/// Check whether the HRESULT represents a failure
pub fn hr_failed(hr: HRESULT) -> bool {
    hr < 0
}

fn create_dxgi_factory_1() -> ComPtr<IDXGIFactory1> {
    unsafe {
        let mut factory = ptr::null_mut();
        let hr = CreateDXGIFactory1(&IID_IDXGIFactory1, &mut factory);
        if hr_failed(hr) {
            panic!("Failed to create DXGIFactory1, {:x}", hr)
        } else {
            ComPtr::from_raw(factory as *mut IDXGIFactory1)
        }
    }
}

fn d3d11_create_device(
    adapter: *mut IDXGIAdapter,
) -> (ComPtr<ID3D11Device>, ComPtr<ID3D11DeviceContext>) {
    unsafe {
        let (mut d3d11_device, mut device_context) = (ptr::null_mut(), ptr::null_mut());
        let hr = D3D11CreateDevice(
            adapter,
            D3D_DRIVER_TYPE_UNKNOWN,
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            0,
            D3D11_SDK_VERSION,
            &mut d3d11_device,
            #[allow(const_item_mutation)]
            &mut D3D_FEATURE_LEVEL_9_1,
            &mut device_context,
        );
        if hr_failed(hr) {
            panic!("Failed to create d3d11 device and device context, {:x}", hr)
        } else {
            (
                ComPtr::from_raw(d3d11_device as *mut ID3D11Device),
                ComPtr::from_raw(device_context),
            )
        }
    }
}

fn get_adapter_outputs(adapter: &IDXGIAdapter1) -> Vec<ComPtr<IDXGIOutput>> {
    let mut outputs = Vec::new();
    for i in 0.. {
        unsafe {
            let mut output = ptr::null_mut();
            if hr_failed(adapter.EnumOutputs(i, &mut output)) {
                break;
            } else {
                let mut out_desc = zeroed();
                (*output).GetDesc(&mut out_desc);
                if out_desc.AttachedToDesktop != 0 {
                    outputs.push(ComPtr::from_raw(output))
                } else {
                    break;
                }
            }
        }
    }
    outputs
}

fn output_is_primary(output: &ComPtr<IDXGIOutput1>) -> bool {
    unsafe {
        let mut output_desc = zeroed();
        output.GetDesc(&mut output_desc);
        let mut monitor_info: MONITORINFO = zeroed();
        monitor_info.cbSize = mem::size_of::<MONITORINFO>() as u32;
        GetMonitorInfoW(output_desc.Monitor, &mut monitor_info);
        (monitor_info.dwFlags & 1) != 0
    }
}

fn get_capture_source(
    output_dups: Vec<(ComPtr<IDXGIOutputDuplication>, ComPtr<IDXGIOutput1>)>,
    cs_index: usize,
) -> Option<(ComPtr<IDXGIOutputDuplication>, ComPtr<IDXGIOutput1>)> {
    if cs_index == 0 {
        output_dups
            .into_iter()
            .find(|(_, out)| output_is_primary(out))
    } else {
        output_dups
            .into_iter()
            .filter(|(_, out)| !output_is_primary(out))
            .nth(cs_index - 1)
    }
}

#[allow(clippy::type_complexity)]
fn duplicate_outputs(
    mut device: ComPtr<ID3D11Device>,
    outputs: Vec<ComPtr<IDXGIOutput>>,
) -> Result<
    (
        ComPtr<ID3D11Device>,
        Vec<(ComPtr<IDXGIOutputDuplication>, ComPtr<IDXGIOutput1>)>,
    ),
    HRESULT,
> {
    let mut out_dups = Vec::new();
    for output in outputs
        .into_iter()
        .map(|out| out.cast::<IDXGIOutput1>().unwrap())
    {
        let dxgi_device = device.up::<IUnknown>();
        let output_duplication = unsafe {
            let mut output_duplication = ptr::null_mut();
            let hr = output.DuplicateOutput(dxgi_device.as_raw(), &mut output_duplication);
            if hr_failed(hr) {
                return Err(hr);
            }
            ComPtr::from_raw(output_duplication)
        };
        device = dxgi_device.cast().unwrap();
        out_dups.push((output_duplication, output));
    }
    Ok((device, out_dups))
}

#[derive(Debug)]
pub enum CursorType {
    Monochrome = 1,
    Color = 2,
    MaskedColor = 4,
}

impl From<u32> for CursorType {
    fn from(value: u32) -> Self {
        match value {
            1 => Self::Monochrome,
            2 => Self::Color,
            4 => Self::MaskedColor,
            _ => unreachable!(),
        }
    }
}

pub struct CursorImage {
    pub data: Vec<u8>,
    pub type_: CursorType,
    pub width: UINT,
    pub height: UINT,
    pub pitch: UINT,
    pub hot_spot: POINT,
}

pub struct Cursor {
    pub position: POINT,
    pub image: Option<CursorImage>,
    pub visible: bool,
}

struct DuplicatedOutput {
    device: ComPtr<ID3D11Device>,
    device_context: ComPtr<ID3D11DeviceContext>,
    output: ComPtr<IDXGIOutput1>,
    output_duplication: ComPtr<IDXGIOutputDuplication>,
}
impl DuplicatedOutput {
    fn get_desc(&self) -> DXGI_OUTPUT_DESC {
        unsafe {
            let mut desc = zeroed();
            self.output.GetDesc(&mut desc);
            desc
        }
    }

    fn capture_cursor_image(
        &mut self,
        size: UINT,
    ) -> Result<(Vec<u8>, DXGI_OUTDUPL_POINTER_SHAPE_INFO), HRESULT> {
        unsafe {
            let mut required_size = 0;
            let mut shape_info = zeroed();
            let mut buffer = vec![0u8; size as usize];

            let hr = self.output_duplication.GetFramePointerShape(
                size,
                buffer.as_mut_ptr() as *mut c_void,
                &mut required_size,
                &mut shape_info,
            );

            if hr_failed(hr) {
                return Err(hr);
            }

            Ok((buffer, shape_info))
        }
    }

    fn capture_frame_to_surface(
        &mut self,
        mut timeout_ms: u32,
        capture_cursor: bool,
    ) -> Result<(ComPtr<IDXGISurface1>, Option<Cursor>), HRESULT> {
        let (frame_resource, frame_info) = unsafe {
            let mut frame_resource = ptr::null_mut();
            let mut frame_info = zeroed();

            // Wait for a vsync event if timeout is 0
            if timeout_ms == 0 {
                self.output.WaitForVBlank();
                timeout_ms = 1000;
            }

            let hr = self.output_duplication.AcquireNextFrame(
                timeout_ms,
                &mut frame_info,
                &mut frame_resource,
            );

            if hr_failed(hr) {
                return Err(hr);
            }
            (ComPtr::from_raw(frame_resource), frame_info)
        };

        // Position information is only accurate when visible is true, otherwise it returns 0,0 position
        //
        // This is part of capture_frame_to_surface because DXGI only allows you to capture the cursor if you "own" the current frame.
        // I tried doing this later in the chain but DXGI returns INVALID_ACCESS
        let cursor = if capture_cursor {
            let mut cursor = Cursor {
                position: frame_info.PointerPosition.Position,
                image: None,
                visible: frame_info.PointerPosition.Visible > 0,
            };

            // if frame_info.PointerShapeBufferSize == 0 then DXGI does not return any image data and is only greater than 0 if the cursor has changed since last frame
            if frame_info.PointerShapeBufferSize > 0 {
                let (buffer, shape) =
                    self.capture_cursor_image(frame_info.PointerShapeBufferSize)?;

                cursor.image = Some(CursorImage {
                    data: buffer,
                    type_: CursorType::from(shape.Type),
                    width: shape.Width,
                    height: shape.Height,
                    pitch: shape.Pitch,
                    hot_spot: shape.HotSpot,
                });
            }

            Some(cursor)
        } else {
            None
        };

        let frame_texture = frame_resource.cast::<ID3D11Texture2D>().unwrap();
        let mut texture_desc = unsafe {
            let mut texture_desc = zeroed();
            frame_texture.GetDesc(&mut texture_desc);
            texture_desc
        };
        // Configure the description to make the texture readable
        texture_desc.Usage = D3D11_USAGE_STAGING;
        texture_desc.BindFlags = 0;
        texture_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        texture_desc.MiscFlags = 0;
        let readable_texture = unsafe {
            let mut readable_texture = ptr::null_mut();
            let hr = self
                .device
                .CreateTexture2D(&texture_desc, ptr::null(), &mut readable_texture);
            if hr_failed(hr) {
                return Err(hr);
            }
            ComPtr::from_raw(readable_texture)
        };
        // Lower priorities causes stuff to be needlessly copied from gpu to ram,
        // causing huge ram usage on some systems.
        unsafe { readable_texture.SetEvictionPriority(DXGI_RESOURCE_PRIORITY_MAXIMUM) };
        let readable_surface = readable_texture.up::<ID3D11Resource>();
        unsafe {
            self.device_context.CopyResource(
                readable_surface.as_raw(),
                frame_texture.up::<ID3D11Resource>().as_raw(),
            );
            self.output_duplication.ReleaseFrame();
        }
        Ok((readable_surface.cast()?, cursor))
    }
}

pub struct BorrowedFrame<'a, 'b> {
    duplicated_output: &'a DuplicatedOutput,
    surface: Option<ComPtr<IDXGISurface1>>,
    pub width: usize,
    pub height: usize,
    data: &'b [BGRA8],
}

#[cfg(feature = "unsafe-send")]
unsafe impl Send for BorrowedFrame<'_, '_> {}

impl Drop for BorrowedFrame<'_, '_> {
    fn drop(&mut self) {
        unsafe {
            if let Some(surface) = &mut self.surface {
                surface.Unmap();
            } else {
                self.duplicated_output
                    .output_duplication
                    .UnMapDesktopSurface();
            }
        }
    }
}

impl<'a> BorrowedFrame<'a, '_> {
    pub fn data(&'a self) -> &'a [BGRA8] {
        self.data
    }
    pub fn as_bytes(&'a self) -> &'a [u8] {
        unsafe {
            ::core::slice::from_raw_parts(
                (self.data as *const [BGRA8]) as *const u8,
                self.width * 4 * self.height,
            )
        }
    }
}

/// Manager of DXGI duplicated outputs
pub struct DXGIManager {
    duplicated_output: Option<DuplicatedOutput>,
    capture_source_index: usize,
    timeout_ms: u32,
}

#[cfg(feature = "unsafe-send")]
unsafe impl Send for DXGIManager {}

struct SharedPtr<T>(*const T);

unsafe impl<T> Send for SharedPtr<T> {}

unsafe impl<T> Sync for SharedPtr<T> {}

impl DXGIManager {
    /// Construct a new manager with capture timeout.
    /// If `timeout_ms` is set to 0, capture will wait until a vsync event.
    pub fn new(timeout_ms: u32) -> Result<DXGIManager, &'static str> {
        let mut manager = DXGIManager {
            duplicated_output: None,
            capture_source_index: 0,
            timeout_ms,
        };

        match manager.acquire_output_duplication() {
            Ok(_) => Ok(manager),
            Err(_) => Err("Failed to acquire output duplication"),
        }
    }

    pub fn geometry(&self) -> (usize, usize) {
        let output_desc = self.duplicated_output.as_ref().unwrap().get_desc();
        let RECT {
            left,
            top,
            right,
            bottom,
        } = output_desc.DesktopCoordinates;
        ((right - left) as usize, (bottom - top) as usize)
    }

    pub fn wait_for_vblank(&self) {
        unsafe {
            self.duplicated_output.as_ref().unwrap().output.WaitForVBlank();
        }
    }

    /// Set index of capture source to capture from
    pub fn set_capture_source_index(&mut self, cs: usize) {
        self.capture_source_index = cs;
        self.acquire_output_duplication().unwrap()
    }

    pub fn get_capture_source_index(&self) -> usize {
        self.capture_source_index
    }

    /// Set timeout to use when capturing
    /// If `timeout_ms` is set to 0, capture will wait until a vsync event.
    pub fn set_timeout_ms(&mut self, timeout_ms: u32) {
        self.timeout_ms = timeout_ms
    }

    /// Duplicate and acquire output selected by `capture_source_index`
    pub fn acquire_output_duplication(&mut self) -> Result<(), &'static str> {
        self.duplicated_output = None;
        let factory = create_dxgi_factory_1();
        for (outputs, adapter) in (0..)
            .map(|i| {
                let mut adapter = ptr::null_mut();
                unsafe {
                    if factory.EnumAdapters1(i, &mut adapter) != DXGI_ERROR_NOT_FOUND {
                        Some(ComPtr::from_raw(adapter))
                    } else {
                        None
                    }
                }
            })
            .take_while(Option::is_some)
            .map(Option::unwrap)
            .map(|adapter| (get_adapter_outputs(&adapter), adapter))
            .filter(|(outs, _)| !outs.is_empty())
        {
            // Creating device for each adapter that has the output
            let (d3d11_device, device_context) = d3d11_create_device(adapter.up().as_raw());
            let (d3d11_device, output_duplications) = duplicate_outputs(d3d11_device, outputs)
                .map_err(|_| "Unable to duplicate output")?;
            if let Some((output_duplication, output)) =
                get_capture_source(output_duplications, self.capture_source_index)
            {
                self.duplicated_output = Some(DuplicatedOutput {
                    device: d3d11_device,
                    device_context,
                    output,
                    output_duplication,
                });
                return Ok(());
            }
        }
        Err("No output could be acquired")
    }

    fn capture_frame_to_surface(
        &mut self,
        capture_cursor: bool,
    ) -> Result<(ComPtr<IDXGISurface1>, Option<Cursor>), CaptureError> {
        if self.duplicated_output.is_none() {
            if self.acquire_output_duplication().is_ok() {
                return Err(CaptureError::Fail("No valid duplicated output"));
            } else {
                return Err(CaptureError::RefreshFailure);
            }
        }
        let timeout_ms = self.timeout_ms;
        match self
            .duplicated_output
            .as_mut()
            .unwrap()
            .capture_frame_to_surface(timeout_ms, capture_cursor)
        {
            Ok((surface, cursor)) => Ok((surface, cursor)),
            Err(DXGI_ERROR_ACCESS_LOST) => {
                if self.acquire_output_duplication().is_ok() {
                    Err(CaptureError::AccessLost)
                } else {
                    Err(CaptureError::RefreshFailure)
                }
            }
            Err(DXGI_ERROR_MORE_DATA) => Err(CaptureError::Fail(
                "Could not create cursor buffer large enough",
            )),
            Err(E_ACCESSDENIED) => Err(CaptureError::AccessDenied),
            Err(DXGI_ERROR_INVALID_CALL) => Err(CaptureError::AccessDenied),
            Err(DXGI_ERROR_WAIT_TIMEOUT) => Err(CaptureError::Timeout),
            Err(_) => {
                if self.acquire_output_duplication().is_ok() {
                    Err(CaptureError::Fail("Failure when acquiring frame"))
                } else {
                    Err(CaptureError::RefreshFailure)
                }
            }
        }
    }

    fn borrow_frame_cpu_t<'a>(
        &'a mut self,
        capture_cursor: bool,
    ) -> Result<(BorrowedFrame<'a, '_>, Option<Cursor>), CaptureError> {
        // This is an optimization that allows us to read the output from system memory if it's already there
        let mapped_surface = unsafe {
            let mut locked_rect = zeroed();
            let hr = self
                .duplicated_output
                .as_mut()
                .unwrap()
                .output_duplication
                .MapDesktopSurface(&mut locked_rect);

            // DXGI_ERROR_UNSUPPORTED means not in system memory
            match hr {
                DXGI_ERROR_UNSUPPORTED => None,
                _ if hr_failed(hr) => {
                    return Err(CaptureError::Fail("Failed to map surface"));
                }
                _ => Some(locked_rect),
            }
        };

        let (frame_surface, mapped_surface, cursor) = match mapped_surface {
            Some(surface) => (None, surface, None),
            None => {
                let (frame_surface, cursor) = match self.capture_frame_to_surface(capture_cursor) {
                    Ok(surface) => surface,
                    Err(e) => return Err(e),
                };

                unsafe {
                    let mut mapped_surface = zeroed();
                    let hr = frame_surface.Map(&mut mapped_surface, DXGI_MAP_READ);
                    if hr_failed(hr) {
                        frame_surface.Release();
                        return Err(CaptureError::Fail("Failed to map surface"));
                    }
                    (Some(frame_surface), mapped_surface, cursor)
                }
            }
        };

        let output_desc = self.duplicated_output.as_mut().unwrap().get_desc();
        let (output_width, output_height) = {
            let RECT {
                left,
                top,
                right,
                bottom,
            } = output_desc.DesktopCoordinates;
            ((right - left) as usize, (bottom - top) as usize)
        };

        let scan_lines = match output_desc.Rotation {
            DXGI_MODE_ROTATION_ROTATE90 | DXGI_MODE_ROTATION_ROTATE270 => output_width,
            _ => output_height,
        };

        let stride = mapped_surface.Pitch as usize / mem::size_of::<BGRA8>();

        let frame = BorrowedFrame {
            duplicated_output: self.duplicated_output.as_ref().unwrap(),
            surface: frame_surface,
            width: output_width,
            height: output_height,
            data: unsafe {
                slice::from_raw_parts(mapped_surface.pBits as *const BGRA8, stride * scan_lines)
            },
        };

        Ok((frame, cursor))
    }

    fn capture_frame_t<T: Copy + Send + Sync + Sized>(
        &mut self,
        capture_cursor: bool,
    ) -> Result<(Vec<T>, (usize, usize), Option<Cursor>), CaptureError> {
        let (frame_surface, cursor) = match self.capture_frame_to_surface(capture_cursor) {
            Ok(surface) => surface,
            Err(e) => return Err(e),
        };

        let mapped_surface = unsafe {
            let mut mapped_surface = zeroed();
            if hr_failed(frame_surface.Map(&mut mapped_surface, DXGI_MAP_READ)) {
                frame_surface.Release();
                return Err(CaptureError::Fail("Failed to map surface"));
            }
            mapped_surface
        };
        let byte_size = |x| x * mem::size_of::<BGRA8>() / mem::size_of::<T>();
        let output_desc = self.duplicated_output.as_mut().unwrap().get_desc();
        let (output_width, output_height) = {
            let RECT {
                left,
                top,
                right,
                bottom,
            } = output_desc.DesktopCoordinates;
            ((right - left) as usize, (bottom - top) as usize)
        };
        // println!(
        //     "output_width: {}, output_height: {}",
        //     output_width, output_height
        // );
        // println!("{:?} {:?}", output_desc.Rotation, output_desc.Monitor);
        let stride = mapped_surface.Pitch as usize / mem::size_of::<BGRA8>();
        // println!("stride: {}", stride);
        let byte_stride = byte_size(stride);

        let scan_lines = match output_desc.Rotation {
            DXGI_MODE_ROTATION_ROTATE90 | DXGI_MODE_ROTATION_ROTATE270 => output_width,
            _ => output_height,
        };

        let mapped_pixels = unsafe {
            slice::from_raw_parts(mapped_surface.pBits as *const T, byte_stride * scan_lines)
        };

        let pixel_buf =
            match output_desc.Rotation {
                DXGI_MODE_ROTATION_IDENTITY | DXGI_MODE_ROTATION_UNSPECIFIED => unsafe {
                    let size = byte_size(output_width * output_height);
                    let mut pixel_buf = Vec::with_capacity(size);
                    let ptr = SharedPtr(pixel_buf.as_ptr() as *const BGRA8);
                    mapped_pixels.par_chunks(byte_stride).enumerate().for_each(
                        |(scan_line, chunk)| {
                            let mut src = chunk.as_ptr() as *const BGRA8;
                            let mut dst = ptr.0 as *mut BGRA8;
                            dst = dst.add(scan_line * output_width);
                            let stop = src.add(output_width);
                            // src = src.add(output_width);
                            while src != stop {
                                src = src.add(1);
                                dst.write(*src);
                                dst = dst.add(1);
                            }
                        },
                    );
                    pixel_buf.set_len(size);
                    pixel_buf
                },
                DXGI_MODE_ROTATION_ROTATE90 => unsafe {
                    let size = byte_size(output_width * output_height);
                    let mut pixel_buf = Vec::with_capacity(size);
                    let ptr = SharedPtr(pixel_buf.as_ptr() as *const BGRA8);
                    mapped_pixels
                        .par_chunks(byte_stride)
                        .rev()
                        .enumerate()
                        .for_each(|(column, chunk)| {
                            let mut src = chunk.as_ptr() as *const BGRA8;
                            let mut dst = ptr.0 as *mut BGRA8;
                            dst = dst.add(column);
                            let stop = src.add(output_height);
                            while src != stop {
                                dst.write(*src);
                                src = src.add(1);
                                dst = dst.add(output_width);
                            }
                        });
                    pixel_buf.set_len(size);
                    pixel_buf
                },
                DXGI_MODE_ROTATION_ROTATE180 => unsafe {
                    let size = byte_size(output_width * output_height);
                    let mut pixel_buf = Vec::with_capacity(size);
                    let ptr = SharedPtr(pixel_buf.as_ptr() as *const BGRA8);
                    mapped_pixels
                        .par_chunks(byte_stride)
                        .rev()
                        .enumerate()
                        .for_each(|(scan_line, chunk)| {
                            let mut src = chunk.as_ptr() as *const BGRA8;
                            let mut dst = ptr.0 as *mut BGRA8;
                            dst = dst.add(scan_line * output_width);
                            let stop = src;
                            src = src.add(output_width);
                            while src != stop {
                                src = src.sub(1);
                                dst.write(*src);
                                dst = dst.add(1);
                            }
                        });
                    pixel_buf.set_len(size);
                    pixel_buf
                },
                DXGI_MODE_ROTATION_ROTATE270 => unsafe {
                    let size = byte_size(output_width * output_height);
                    let mut pixel_buf = Vec::with_capacity(size);
                    let ptr = SharedPtr(pixel_buf.as_ptr() as *const BGRA8);
                    mapped_pixels.par_chunks(byte_stride).enumerate().for_each(
                        |(column, chunk)| {
                            let mut src = chunk.as_ptr() as *const BGRA8;
                            let mut dst = ptr.0 as *mut BGRA8;
                            dst = dst.add(column);
                            let stop = src;
                            src = src.add(output_height);
                            while src != stop {
                                src = src.sub(1);
                                dst.write(*src);
                                dst = dst.add(output_width);
                            }
                        },
                    );
                    pixel_buf.set_len(size);
                    pixel_buf
                },
                n => unreachable!("Undefined DXGI_MODE_ROTATION: {}", n),
            };
        // println!(
        //     "pixel_buf.cap: {} ; pixel_buf.len: {} ; expected {}",
        //     pixel_buf.capacity(),
        //     pixel_buf.len(),
        //     output_width * output_height * (mem::size_of::<BGRA8>() / mem::size_of::<T>())
        // );
        unsafe { frame_surface.Unmap() };
        Ok((pixel_buf, (output_width, output_height), cursor))
    }

    /// Capture a frame
    ///
    /// On success, return Vec with pixels and width and height of frame.
    /// On failure, return CaptureError.
    pub fn capture_frame(
        &mut self,
        capture_cursor: bool,
    ) -> Result<(Vec<BGRA8>, (usize, usize), Option<Cursor>), CaptureError> {
        self.capture_frame_t(capture_cursor)
    }

    /// Capture a frame
    ///
    /// On success, return Vec with pixel components and width and height of frame.
    /// On failure, return CaptureError.
    pub fn capture_frame_components(
        &mut self,
        capture_cursor: bool,
    ) -> Result<(Vec<u8>, (usize, usize), Option<Cursor>), CaptureError> {
        self.capture_frame_t(capture_cursor)
    }

    // TODO: replace with gpu implementation
    pub fn capture_frame_rgba(
        &mut self,
        capture_cursor: bool,
    ) -> Result<(Vec<RGBA8>, (usize, usize), Option<Cursor>), CaptureError> {
        let (mut frame, size, cursor) = self.capture_frame(capture_cursor)?;
        frame.par_iter_mut().for_each(|px| {
            ::std::mem::swap(&mut px.b, &mut px.r);
        });
        let frame = unsafe { ::std::mem::transmute(frame) };
        Ok((frame, size, cursor))
    }

    /// Borrow a frame without allocations
    pub fn borrow_frame(
        &mut self,
        capture_cursor: bool,
    ) -> Result<(BorrowedFrame, Option<Cursor>), CaptureError> {
        self.borrow_frame_cpu_t(capture_cursor)
    }

    // TODO: replace with gpu implementation
    pub fn capture_frame_rgba_components(
        &mut self,
        capture_cursor: bool,
    ) -> Result<(Vec<u8>, (usize, usize), Option<Cursor>), CaptureError> {
        let (mut frame, size, cursor) = self.capture_frame_components(capture_cursor)?;

        frame.par_chunks_exact_mut(4).for_each(|px| {
            px.swap(0, 2);
        });
        Ok((frame, size, cursor))
    }
}

#[cfg(test)]
mod dxgcap_tests {
    use serial_test::serial;

    use super::*;

    #[test]
    #[serial]
    fn cap_100_frames() {
        let mut manager = DXGIManager::new(300).unwrap();
        for _ in 0..10 {
            match manager.capture_frame(false) {
                Ok((pixels, (_, _), _)) => {
                    let len = pixels.len() as u64;
                    let (r, g, b) = pixels.into_iter().fold((0u64, 0u64, 0u64), |(r, g, b), p| {
                        (r + p.r as u64, g + p.g as u64, b + p.b as u64)
                    });
                    println!("avg: {} {} {}", r / len, g / len, b / len);
                    println!("")
                }
                Err(e) => println!("error: {:?}", e),
            }
        }
    }

    #[test]
    #[serial]
    fn compare_frame_dims() {
        let mut manager = DXGIManager::new(300).unwrap();
        let (frame, (fw, fh), _) = manager.capture_frame(false).unwrap();
        let (frame_u8, (fu8w, fu8h), _) = manager.capture_frame_components(false).unwrap();
        assert_eq!(fw, fu8w);
        assert_eq!(fh, fu8h);
        assert_eq!(4 * frame.len(), frame_u8.len());
        assert_eq!(fw * fh, frame.len());
    }
}
