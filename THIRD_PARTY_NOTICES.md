# Third-party components and model provenance

This project uses, but does not claim ownership of, these upstream components:

- [MMPose / RTMPose](https://github.com/open-mmlab/mmpose): Apache-2.0 code. Published body7 RTMPose-M ONNX checkpoint; original model training/dataset conditions remain relevant.
- [RTMLib](https://github.com/Tau-J/rtmlib): Apache-2.0. Lightweight pose-inference adapter.
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX): Apache-2.0 code. Person detector checkpoint distributed by OpenMMLab.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime): MIT.
- [OpenCV](https://github.com/opencv/opencv): Apache-2.0 for the selected version.
- [FastAPI](https://github.com/fastapi/fastapi), [Starlette](https://github.com/encode/starlette), [Uvicorn](https://github.com/encode/uvicorn), [React](https://github.com/facebook/react), [Three.js](https://github.com/mrdoob/three.js), [Vite](https://github.com/vitejs/vite): retain their respective upstream license notices when redistributing dependencies.
- [Lucide](https://github.com/lucide-icons/lucide): ISC icons.

The package manager records dependency versions and distributes dependency licenses with packages. The project contains model download URLs and verified hashes, not model binaries. No model was trained or fine-tuned as part of this implementation.

The public RTMLib `demo.jpg` was downloaded only into the separate local testing directory for an inference smoke test. It is not included in this repository or presented as captured multi-camera data. The in-app synthetic images are generated from calibrated geometric fixtures and are labelled synthetic.

The license for the newly authored application source has not been selected on the repository owner's behalf.

## Real-time desktop app

- [three.js](https://github.com/mrdoob/three.js) r180 is vendored in `backend/realtime/ui/vendor/` under the MIT license (`THREE_LICENSE.txt`).
- [pywebview](https://github.com/r0x0r/pywebview) (BSD-3-Clause) provides the native window when installed from `requirements-desktop.txt`.
- The optional RTMPose-X and Halpe-26 (feet) checkpoints are downloaded from OpenMMLab on request. Review their training-data terms before commercial use.
- The demo in `docs/media/` shows the public-domain NASA astronaut portrait distributed with scikit-image (`skimage.data.astronaut`), mapped onto a synthetic moving plane.
