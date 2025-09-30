# Design Document — Gradio Interface for Hunyuan3D-Omni

## Purpose

Provide a **unified graphical interface** for multimodal 3D inference using Tencent’s Hunyuan3D-Omni system. The Gradio app should:

* Allow users to configure and run inference in **bbox**, **pose**, **point**, and **voxel** control modes.
* Dynamically adjust input panels based on selected control type.
* Provide post-processing options.
* Display results using a **3D model viewer** before exporting.
* Support **presets** for bounding boxes and bone structures.

---

## Application Structure

```
+-------------------------------------------+
| Hunyuan3D-Omni Inference Interface        |
+-------------------------------------------+
| [ Control Type: Dropdown ]                |
| [ Common Inference Settings Panel ]       |
|   └── GPU ID                              |
|   └── # GPUs                              |
|   └── Save Directory                      |
|   └── Model Repo ID                       |
|   └── [x] Use EMA                         |
|   └── [x] Use FlashVDM                    |
|                                           |
| [ Conditional Panel (bbox / pose / ... ]  |
|                                           |
| [ Post-Processing Toggles ]               |
|   └── [x] Remove Floaters                 |
|   └── [x] Remove Degenerate Faces         |
|                                           |
| [ Run Inference Button ]                  |
|                                           |
| [ Output Viewer ]                         |
|   └── 3D model preview                    |
|   └── Image preview                       |
|   └── Download options (.glb, .ply, .zip) |
+-------------------------------------------+
```

---

## UI Sections in Detail

### 1. **Control Type Selector**

* `Dropdown`: `bbox`, `pose`, `point`, `voxel`
* Dynamically reveals control-specific UI below

---

### 2. **Common Settings Panel**

| Field            | Type                        | Notes                               |
| ---------------- | --------------------------- | ----------------------------------- |
| GPU ID           | Number                      | Default: `0`                        |
| Number of GPUs   | Number                      | Default: `1`                        |
| Output Directory | Textbox / Filepath selector | Default: `./omni_inference_results` |
| Model Repo ID    | Textbox                     | Default: `tencent/Hunyuan3D-Omni`   |
| Use EMA          | Checkbox                    | Toggle `--use_ema`                  |
| Use FlashVDM     | Checkbox                    | Toggle `--flashvdm`                 |

---

### 3. **Control-Specific Panels**

---

#### **Bounding Box Control**

* [ ] **Upload Image Files**: `Image[]`
* [ ] **Upload or Input Bounding Boxes**:

  * Option 1: Upload `data.json`
  * Option 2: Manually enter JSON textbox
  * Option 3: Select preset from `bbox_presets/`
* [ ] **Optional Preset Selector** (dropdown from `bbox_presets/`)

---

#### **Pose Control**

* [ ] **Upload Image Files**: `Image[]`
* [ ] **Upload Bone Structure Files** (`.txt`)
* [ ] **Optional Preset Selector** (dropdown from `bone_presets/`)
* [ ] **Pose Name Mapping** (optional renaming)

---

#### **Point Cloud Control**

* [ ] **Upload Image Files**: `Image[]`
* [ ] **Upload Point Cloud Files**: `.ply`, `.obj`
* [ ] **Optional JSON Upload**: full `{ "image": [...], "point": [...] }`

---

#### **Voxel Control**

* [ ] **Upload Image Files**: `Image[]`
* [ ] **Upload Voxel Mesh Files**: `.ply`, `.obj`
* [ ] **Optional JSON Upload**: full `{ "image": [...], "voxel": [...] }`

---

### 4. **Post-Processing Toggles**

| Option                      | Description                     |
| --------------------------- | ------------------------------- |
| [x] Remove Floaters         | Apply `FloaterRemover()`        |
| [x] Remove Degenerate Faces | Apply `DegenerateFaceRemover()` |

Both enabled by default.

---

### 5. **Run Button**

* Begins inference pipeline
* Displays progress per image
* Disables UI inputs while running

---

### 6. **Results Viewer**

**For each output:**

* Display:

  * Input image preview
  * Download links for:

    * `.glb` mesh
    * `.ply` point cloud

* View 3D Output:

  * Use [`three.js`](https://threejs.org/) via Gradio HTML for `.glb`
  * Optional: use `pyvista`, `trimesh`, or `plotly` for embedded viewer

* Final Output:

  * [Download All as ZIP]

---

## Preset Handling

### `bbox_presets/`

* JSON files with keys `"image"` and `"bbox"`
* Populate a dropdown in bbox mode:

  * “car_bbox”, “person_bbox”, etc.
* On selection: auto-fill image + bbox fields

### `bone_presets/`

* `.txt` files with pose bone structure
* Populate dropdown in pose mode:

  * “A-pose”, “T-pose”, “hands-up”
* Use filename as pose name key

---

## Backend Considerations

* Use `tempfile.TemporaryDirectory()` for file I/O isolation
* Normalize inputs per `trimesh` expectations
* Guard against malformed inputs
* Support asynchronous processing
* Handle `.glb`, `.ply`, `.png` export in same dir
* Return output file paths for display & download

---

## Internationalization

* All tooltips, labels, and UI text should be in **English**
* Translate existing Chinese warnings (e.g., "文件路径不存在")
