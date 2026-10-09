# Neo-3D asset destinations

- `models/`: imported glTF 2.0 `.gltf` and binary `.glb` files.
- `textures/`: source images (PNG/JPEG/KTX2 when a decoder is integrated).
- `materials/`: future serialized material definitions.
- `scenes/`: scene files and entity data.

Keep original imported files intact. A model importer must validate glTF buffers, accessors, indices, material references and texture paths before creating GPU resources. The current native viewport does not yet load these assets; this directory establishes the source asset contract.
