(() => {
  "use strict";

  const MODEL_SIZE = 256;
  const MAX_IMAGE_SIZE = 20 * 1024 * 1024;
  const MAX_RENDER_EDGE = 1400;
  const SAMPLE_MODEL_URLS = {
    semantic: "./models/lraspp-voc21.onnx",
    medical: "./models/unet-hippocampus.onnx",
  };
  const SAMPLE_IMAGE_URL = "./samples/hippocampus-mri.png";
  const MEDICAL_CLASSES = ["background", "anterior", "posterior"];
  const VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike",
    "person", "potted plant", "sheep", "sofa", "train", "tv",
  ];
  const PALETTE = [
    [32, 39, 33], [91, 120, 255], [204, 255, 56], [255, 107, 61],
    [142, 92, 246], [25, 183, 165], [255, 195, 46], [237, 73, 125],
    [76, 166, 74], [43, 197, 238], [248, 137, 205], [125, 104, 69],
    [247, 89, 65], [67, 88, 190], [152, 207, 85], [246, 160, 70],
    [150, 93, 167], [40, 139, 123], [201, 178, 54], [174, 65, 96], [84, 124, 100],
  ];

  const elements = {
    modelInput: document.querySelector("#modelInput"),
    modelChooseButton: document.querySelector("#modelChooseButton"),
    modelDrop: document.querySelector("#modelDrop"),
    modelName: document.querySelector("#modelName"),
    modelProgress: document.querySelector("#modelProgress"),
    sampleModelButton: document.querySelector("#sampleModelButton"),
    imageInput: document.querySelector("#imageInput"),
    imageChooseButton: document.querySelector("#imageChooseButton"),
    sampleImageButton: document.querySelector("#sampleImageButton"),
    imageDrop: document.querySelector("#imageDrop"),
    imageName: document.querySelector("#imageName"),
    classNames: document.querySelector("#classNames"),
    runButton: document.querySelector("#runButton"),
    setupMessage: document.querySelector("#setupMessage"),
    stage: document.querySelector("#stage"),
    resultCanvas: document.querySelector("#resultCanvas"),
    inputCanvas: document.querySelector("#inputCanvas"),
    emptyState: document.querySelector("#emptyState"),
    busyState: document.querySelector("#busyState"),
    downloadButton: document.querySelector("#downloadButton"),
    metricModel: document.querySelector("#metricModel"),
    metricOutput: document.querySelector("#metricOutput"),
    metricRuntime: document.querySelector("#metricRuntime"),
    metricTiming: document.querySelector("#metricTiming"),
    legend: document.querySelector("#legend"),
    activeClassCount: document.querySelector("#activeClassCount"),
    viewTabs: [...document.querySelectorAll(".view-tab")],
    trackButtons: [...document.querySelectorAll(".track-button")],
  };

  const state = {
    session: null,
    modelName: "",
    image: null,
    imageName: "",
    predictions: null,
    outputWidth: 0,
    outputHeight: 0,
    classCount: 0,
    counts: [],
    view: "overlay",
    track: "semantic",
  };

  const inputContext = elements.inputCanvas.getContext("2d", { willReadFrequently: true });
  const resultContext = elements.resultCanvas.getContext("2d");

  function setMessage(message, isError = false) {
    elements.setupMessage.textContent = message;
    elements.setupMessage.style.color = isError ? "#b93421" : "";
  }

  function setModelBusy(isBusy) {
    elements.modelProgress.hidden = !isBusy;
    elements.sampleModelButton.disabled = isBusy;
    elements.modelChooseButton.disabled = isBusy;
  }

  function setInferenceBusy(isBusy) {
    elements.busyState.hidden = !isBusy;
    elements.runButton.disabled = isBusy || !state.session || !state.image;
  }

  function updateRunState() {
    elements.runButton.disabled = !state.session || !state.image;
  }

  function getClassNames() {
    return elements.classNames.value
      .split(",")
      .map((name) => name.trim())
      .filter(Boolean);
  }

  async function createSession(modelSource, displayName) {
    if (!window.ort) throw new Error("ONNX Runtime Web을 불러오지 못했습니다.");
    setModelBusy(true);
    setMessage("ONNX graph와 연산자를 확인하는 중입니다.");
    try {
      window.ort.env.wasm.wasmPaths = new URL("./vendor/", window.location.href).href;
      window.ort.env.wasm.numThreads = window.crossOriginIsolated
        ? Math.min(navigator.hardwareConcurrency || 2, 4)
        : 1;
      state.session = await window.ort.InferenceSession.create(modelSource, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      state.modelName = displayName;
      elements.modelName.textContent = displayName;
      elements.modelName.title = displayName;
      elements.modelDrop.classList.add("is-ready");
      elements.metricModel.textContent = displayName;
      elements.metricModel.title = displayName;
      setMessage(`모델 준비 완료 · input: ${state.session.inputNames[0]}`);
      updateRunState();
    } catch (error) {
      console.error(error);
      state.session = null;
      elements.modelDrop.classList.remove("is-ready");
      setMessage(`모델을 열지 못했습니다: ${error.message}`, true);
      throw error;
    } finally {
      setModelBusy(false);
    }
  }

  async function loadSampleModel() {
    if (state.track === "medical") {
      elements.classNames.value = MEDICAL_CLASSES.join(",");
      await createSession(SAMPLE_MODEL_URLS.medical, "Tiny U-Net · Hippocampus 3");
      return;
    }
    elements.classNames.value = VOC_CLASSES.join(",");
    await createSession(SAMPLE_MODEL_URLS.semantic, "LRASPP · VOC 21");
  }

  async function loadLocalModel(file) {
    if (!file || !file.name.toLowerCase().endsWith(".onnx")) {
      setMessage(".onnx 모델 파일을 선택해 주세요.", true);
      return;
    }
    const buffer = await file.arrayBuffer();
    await createSession(new Uint8Array(buffer), file.name);
  }

  function setImageSource(url, displayName, shouldRevoke = false) {
    const image = new Image();
    image.onload = () => {
      if (shouldRevoke) URL.revokeObjectURL(url);
      state.image = image;
      state.imageName = displayName;
      elements.imageName.textContent = `${displayName} · ${image.naturalWidth}×${image.naturalHeight}`;
      elements.imageName.title = `${displayName} · ${image.naturalWidth}×${image.naturalHeight}`;
      elements.imageDrop.classList.add("is-ready");
      setMessage(state.session ? "실행할 준비가 됐습니다." : "이제 ONNX 모델을 준비하세요.");
      updateRunState();
    };
    image.onerror = () => {
      if (shouldRevoke) URL.revokeObjectURL(url);
      setMessage("이미지를 읽지 못했습니다.", true);
    };
    image.src = url;
  }

  function loadImage(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setMessage("이미지 파일만 사용할 수 있습니다.", true);
      return;
    }
    if (file.size > MAX_IMAGE_SIZE) {
      setMessage("20 MB 이하 이미지를 선택해 주세요.", true);
      return;
    }
    setImageSource(URL.createObjectURL(file), file.name, true);
  }

  function loadSampleImage() {
    setTrack("medical");
    setImageSource(SAMPLE_IMAGE_URL, "hippocampus_252_z12.png");
  }

  function preprocessImage() {
    inputContext.clearRect(0, 0, MODEL_SIZE, MODEL_SIZE);
    inputContext.drawImage(state.image, 0, 0, MODEL_SIZE, MODEL_SIZE);
    const pixels = inputContext.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
    const tensor = new Float32Array(3 * MODEL_SIZE * MODEL_SIZE);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    const plane = MODEL_SIZE * MODEL_SIZE;
    for (let pixelIndex = 0; pixelIndex < plane; pixelIndex += 1) {
      const sourceIndex = pixelIndex * 4;
      tensor[pixelIndex] = (pixels[sourceIndex] / 255 - mean[0]) / std[0];
      tensor[plane + pixelIndex] = (pixels[sourceIndex + 1] / 255 - mean[1]) / std[1];
      tensor[plane * 2 + pixelIndex] = (pixels[sourceIndex + 2] / 255 - mean[2]) / std[2];
    }
    return new window.ort.Tensor("float32", tensor, [1, 3, MODEL_SIZE, MODEL_SIZE]);
  }

  function decodeOutput(output) {
    const dims = output.dims.map(Number);
    if (dims.length !== 4 || dims[0] !== 1) {
      throw new Error(`출력은 [1,C,H,W]여야 합니다. 현재: [${dims.join(",")}]`);
    }
    const [, classCount, height, width] = dims;
    if (classCount < 2 || height < 2 || width < 2) {
      throw new Error(`유효하지 않은 출력 shape입니다: [${dims.join(",")}]`);
    }
    const plane = height * width;
    const predictions = new Uint16Array(plane);
    const counts = new Array(classCount).fill(0);
    for (let pixelIndex = 0; pixelIndex < plane; pixelIndex += 1) {
      let bestClass = 0;
      let bestScore = output.data[pixelIndex];
      for (let classIndex = 1; classIndex < classCount; classIndex += 1) {
        const score = output.data[classIndex * plane + pixelIndex];
        if (score > bestScore) {
          bestScore = score;
          bestClass = classIndex;
        }
      }
      predictions[pixelIndex] = bestClass;
      counts[bestClass] += 1;
    }
    state.predictions = predictions;
    state.outputWidth = width;
    state.outputHeight = height;
    state.classCount = classCount;
    state.counts = counts;
    return dims;
  }

  function resolvedClassNames() {
    const names = getClassNames();
    return Array.from(
      { length: state.classCount },
      (_, index) => names[index] || `class ${index}`,
    );
  }

  function renderResult() {
    if (!state.predictions || !state.image) return;
    const scale = Math.min(
      1,
      MAX_RENDER_EDGE / Math.max(state.image.naturalWidth, state.image.naturalHeight),
    );
    const width = Math.max(1, Math.round(state.image.naturalWidth * scale));
    const height = Math.max(1, Math.round(state.image.naturalHeight * scale));
    elements.resultCanvas.width = width;
    elements.resultCanvas.height = height;
    resultContext.imageSmoothingEnabled = true;
    resultContext.clearRect(0, 0, width, height);

    if (state.view !== "mask") {
      resultContext.drawImage(state.image, 0, 0, width, height);
    }
    if (state.view === "source") return;

    const maskCanvas = document.createElement("canvas");
    maskCanvas.width = state.outputWidth;
    maskCanvas.height = state.outputHeight;
    const maskContext = maskCanvas.getContext("2d");
    const maskImage = maskContext.createImageData(state.outputWidth, state.outputHeight);
    for (let index = 0; index < state.predictions.length; index += 1) {
      const classIndex = state.predictions[index];
      const color = PALETTE[classIndex % PALETTE.length];
      const offset = index * 4;
      maskImage.data[offset] = color[0];
      maskImage.data[offset + 1] = color[1];
      maskImage.data[offset + 2] = color[2];
      maskImage.data[offset + 3] = state.view === "overlay" && classIndex === 0 ? 0 : 235;
    }
    maskContext.putImageData(maskImage, 0, 0);
    resultContext.save();
    resultContext.imageSmoothingEnabled = false;
    resultContext.globalAlpha = state.view === "overlay" ? 0.62 : 1;
    resultContext.drawImage(maskCanvas, 0, 0, width, height);
    resultContext.restore();
  }

  function renderLegend() {
    const names = resolvedClassNames();
    const total = state.counts.reduce((sum, value) => sum + value, 0);
    const active = state.counts.filter((count) => count > 0).length;
    elements.activeClassCount.textContent = `${active} ACTIVE`;
    elements.legend.className = "legend-items";
    elements.legend.innerHTML = "";
    state.counts
      .map((count, index) => ({ count, index, ratio: total ? count / total : 0 }))
      .filter((item) => item.count > 0)
      .sort((a, b) => b.count - a.count)
      .forEach(({ count, index, ratio }) => {
        const item = document.createElement("div");
        item.className = "legend-item";
        const swatch = document.createElement("i");
        swatch.className = "legend-swatch";
        swatch.style.backgroundColor = `rgb(${PALETTE[index % PALETTE.length].join(",")})`;
        const name = document.createElement("span");
        name.className = "legend-name";
        name.textContent = `${index} · ${names[index]}`;
        const value = document.createElement("span");
        value.className = "legend-value";
        value.textContent = `${(ratio * 100).toFixed(1)}% · ${count.toLocaleString()}`;
        item.append(swatch, name, value);
        elements.legend.append(item);
      });
  }

  async function runInference() {
    if (!state.session || !state.image) return;
    setInferenceBusy(true);
    setMessage("브라우저에서 추론 중입니다.");
    try {
      const input = preprocessImage();
      const inputName = state.session.inputNames[0];
      const start = performance.now();
      const outputs = await state.session.run({ [inputName]: input });
      const elapsed = performance.now() - start;
      const outputName = state.session.outputNames[0];
      const dims = decodeOutput(outputs[outputName]);
      elements.metricOutput.textContent = dims.join("×");
      elements.metricTiming.textContent = `${elapsed.toFixed(0)} ms`;
      elements.stage.classList.remove("is-empty");
      elements.emptyState.hidden = true;
      elements.downloadButton.disabled = false;
      renderResult();
      renderLegend();
      const labelCount = getClassNames().length;
      setMessage(
        labelCount === state.classCount
          ? `${state.classCount}개 클래스 예측 완료.`
          : `출력은 ${state.classCount}개 클래스지만 이름은 ${labelCount}개입니다. 누락 이름은 자동 생성했습니다.`,
        labelCount !== state.classCount,
      );
    } catch (error) {
      console.error(error);
      setMessage(`추론 실패: ${error.message}`, true);
    } finally {
      setInferenceBusy(false);
    }
  }

  function setView(view) {
    state.view = view;
    elements.viewTabs.forEach((tab) => {
      const active = tab.dataset.view === view;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    renderResult();
  }

  function updateTrackButtons() {
    elements.trackButtons.forEach((button) => {
      const active = button.dataset.track === state.track;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
  }

  function setTrack(track) {
    state.track = track;
    updateTrackButtons();
    if (track === "medical") {
      elements.classNames.value = MEDICAL_CLASSES.join(",");
      elements.sampleModelButton.children[0].textContent = "학습된 U-Net 예제 사용";
      elements.sampleModelButton.children[1].textContent = "4.34 MB ↓";
      setMessage("MSD Hippocampus U-Net을 선택하세요. 회색조 입력은 RGB로 복제됩니다.");
    } else {
      elements.classNames.value = VOC_CLASSES.join(",");
      elements.sampleModelButton.children[0].textContent = "공식 LRASPP 예제 사용";
      elements.sampleModelButton.children[1].textContent = "12.9 MB ↓";
      setMessage("다중 클래스 LRASPP 모델을 선택하거나 공식 예제를 사용하세요.");
    }
  }

  function bindDropZone(zone, callback) {
    ["dragenter", "dragover"].forEach((eventName) => {
      zone.addEventListener(eventName, (event) => {
        event.preventDefault();
        zone.classList.add("is-dragging");
      });
    });
    ["dragleave", "drop"].forEach((eventName) => {
      zone.addEventListener(eventName, (event) => {
        event.preventDefault();
        zone.classList.remove("is-dragging");
      });
    });
    zone.addEventListener("drop", (event) => callback(event.dataTransfer.files[0]));
  }

  elements.modelChooseButton.addEventListener("click", () => elements.modelInput.click());
  elements.imageChooseButton.addEventListener("click", () => elements.imageInput.click());
  elements.modelInput.addEventListener("change", (event) => loadLocalModel(event.target.files[0]));
  elements.imageInput.addEventListener("change", (event) => loadImage(event.target.files[0]));
  elements.sampleImageButton.addEventListener("click", loadSampleImage);
  elements.sampleModelButton.addEventListener("click", loadSampleModel);
  elements.runButton.addEventListener("click", runInference);
  elements.viewTabs.forEach((tab) => tab.addEventListener("click", () => setView(tab.dataset.view)));
  elements.trackButtons.forEach((button) => button.addEventListener("click", () => setTrack(button.dataset.track)));
  elements.downloadButton.addEventListener("click", () => {
    if (elements.downloadButton.disabled) return;
    const link = document.createElement("a");
    link.download = `${state.imageName.replace(/\.[^.]+$/, "") || "segmentation"}-${state.view}.png`;
    link.href = elements.resultCanvas.toDataURL("image/png");
    link.click();
  });
  bindDropZone(elements.modelDrop, loadLocalModel);
  bindDropZone(elements.imageDrop, loadImage);
})();
