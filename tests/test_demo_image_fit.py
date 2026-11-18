"""Offline geometry, mask alignment and preview provenance contracts."""
import numpy as np
from PIL import Image
import pytest

from diffusion_editor.demo.image_fit import ImageFit, prepare_upload, resolve_fitted_mask
from diffusion_editor.demo.inputs import editor_canvas


@pytest.mark.parametrize("size,content,canvas", [
    ((512, 768), (512, 768), (512, 768)),
    ((513, 769), (513, 769), (520, 776)),
    ((1501, 1001), (1024, 683), (1024, 688)),
    ((1001, 1501), (683, 1024), (688, 1024)),
    ((1, 2), (1, 2), (8, 8)),
    ((10000, 1), (1024, 1), (1024, 8)),
])
def test_geometry_without_crop_or_small_image_upscale(size, content, canvas):
    fit = ImageFit.plan(size)
    assert fit.content_size == content and fit.canvas_size == canvas
    assert all(x % 8 == 0 and x <= 1024 for x in canvas)
    assert fit.changed == (size != content or content != canvas)


def test_padding_replicates_edges_and_retains_every_source_pixel():
    pixels = np.arange(13 * 17 * 3, dtype=np.uint8).reshape(13, 17, 3)
    result = prepare_upload(Image.fromarray(pixels))
    actual = np.asarray(result.image)
    np.testing.assert_array_equal(actual[:13, :17], pixels)
    np.testing.assert_array_equal(actual[:13, 17:], np.repeat(pixels[:, -1:], 7, axis=1))
    np.testing.assert_array_equal(actual[13:], np.repeat(actual[12:13], 3, axis=0))
    assert result.fit.canvas_size == (24, 16)


def test_noop_pixels_and_disabled_fitting():
    image = Image.new("RGB", (24, 16), "red")
    result = prepare_upload(image)
    assert result.image.tobytes() == image.tobytes() and not result.fit.changed
    strict = prepare_upload(Image.new("RGB", (1501, 1001)), enabled=False)
    assert strict.image.size == (1501, 1001) and not strict.fit.changed


@pytest.mark.parametrize("size,kwargs", [((0, 10), {}), ((True, 10), {}), ((10, 10), {"alignment": 0}),
                                         ((10, 10), {"max_side": 4}), ((10, 10), {"enabled": 1})])
def test_invalid_geometry(size, kwargs):
    with pytest.raises(ValueError):
        ImageFit.plan(size, **kwargs)


def test_original_content_and_canvas_masks_share_geometry():
    fit = ImageFit.plan((1501, 1001))
    for size, space in ((fit.source_size, "original_upload"), (fit.content_size, "resized_content"),
                        (fit.canvas_size, "model_canvas")):
        mask, info = fit.mask(Image.new("L", size, 255), binary=True)
        actual = np.asarray(mask)
        assert mask.size == fit.canvas_size and info["coordinate_space"] == space
        assert (actual[:683] == 255).all() and (actual[683:] == 0).all()
    with pytest.raises(ValueError, match="not aligned"):
        fit.mask(Image.new("L", (50, 60)), binary=True)
    with pytest.raises(ValueError, match="grayscale"):
        fit.mask(Image.new("RGB", fit.source_size), binary=True)


def test_soft_and_binary_mask_resizing():
    fit = ImageFit.plan((1031, 31))
    raw = Image.new("L", fit.source_size)
    raw.paste(255, (515, 0, 1031, 31))
    binary, binary_info = fit.mask(raw, binary=True)
    soft, soft_info = fit.mask(raw, binary=False)
    assert set(np.unique(binary)) == {0, 255}
    assert ((np.asarray(soft) > 0) & (np.asarray(soft) < 255)).any()
    assert binary_info["resize_filter"] == "nearest" and soft_info["resize_filter"] == "bilinear"
    assert (np.asarray(soft)[fit.content_size[1]:] == 0).all()


def test_state_reuses_source_restores_original_and_rejects_stale_preview():
    raw = Image.new("RGB", (17, 13), "red")
    fitted = prepare_upload(raw)
    assert prepare_upload(fitted.image, fitted) is fitted
    strict = prepare_upload(fitted.image, fitted, enabled=False)
    assert strict.image.size == raw.size and strict.image.tobytes() == raw.tobytes()
    again = prepare_upload(strict.image, strict)
    assert again.fit == fitted.fit and again.image.tobytes() == fitted.image.tobytes()
    replaced = prepare_upload(Image.new("RGB", fitted.image.size, "blue"), fitted)
    assert replaced.original.size == fitted.image.size
    assert replaced.original_digest != fitted.original_digest


@pytest.mark.parametrize("use_original", [False, True])
def test_draw_on_original_or_canvas_remains_aligned(use_original):
    prepared = prepare_upload(Image.new("RGB", (17, 13), "red"))
    background = prepared.original if use_original else prepared.image
    drawing = editor_canvas(background)
    layer = Image.new("RGBA", background.size)
    layer.putpixel((3, 4), (255, 255, 255, 128))
    drawing["layers"] = [layer]
    mask, raw, info = resolve_fitted_mask(prepared, "Draw", None, drawing, binary=False)
    assert mask.size == (24, 16) and mask.getpixel((3, 4)) == 128
    assert np.count_nonzero(mask) == 1
    assert info["coordinate_space"] == ("original_upload" if use_original else "model_canvas")


def test_draw_rejects_stale_background_and_padding_only_paint():
    prepared = prepare_upload(Image.new("RGB", (17, 13), "red"))
    stale = editor_canvas(Image.new("RGB", prepared.image.size, "blue"))
    with pytest.raises(ValueError, match="different image"):
        resolve_fitted_mask(prepared, "Draw", None, stale, binary=True)
    drawing = editor_canvas(prepared.image)
    layer = Image.new("RGBA", prepared.image.size)
    layer.putpixel((20, 14), (255, 255, 255, 255))
    drawing["layers"] = [layer]
    with pytest.raises(ValueError, match="inside the image content"):
        resolve_fitted_mask(prepared, "Draw", None, drawing, binary=True)


def test_exif_orientation_precedes_fitting():
    image = Image.new("RGB", (13, 17), "red")
    image.getexif()[274] = 6
    fitted = prepare_upload(image)
    assert fitted.original.size == (17, 13) and fitted.image.size == (24, 16)
