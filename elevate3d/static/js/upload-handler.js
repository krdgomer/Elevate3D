document.getElementById("upload-btn").addEventListener("click", async () => {
  const fileInput = document.getElementById("file-input");
  if (!fileInput.files.length) {
    alert("Please select an image file first");
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);

  document.getElementById("loading").style.display = "block";

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
      headers: {
        Accept: "application/json",
      },
    });

    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ error: "Unknown error" }));
      throw new Error(error.message || error.error || "Server error");
    }

    const data = await response.json();
    if (!data.model_url) {
      throw new Error("Invalid response from server");
    }

    loadModel(data.model_url);
  } catch (error) {
    console.error("Upload error:", error);
    alert(`Error: ${error.message}`);
  } finally {
    document.getElementById("loading").style.display = "none";
  }
});

function loadModel(url) {
  while (scene.children.length > 4) {
    scene.remove(scene.children[4]);
  }

  const loader = new THREE.GLTFLoader();

  loader.load(
    url,
    (gltf) => {
      const model = gltf.scene;

      model.rotation.x = -Math.PI / 2;

      model.traverse((child) => {
        if (child.isMesh) {
          child.material.side = THREE.DoubleSide;
          if (child.material.map) {
            child.material.map.encoding = THREE.sRGBEncoding;
          }
          child.material.needsUpdate = true;
        }
      });

      const box = new THREE.Box3().setFromObject(model);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());

      model.position.sub(center);
      const maxDim = Math.max(size.x, size.y, size.z);
      const scale = 1.5 / maxDim;
      model.scale.set(scale, scale, scale);

      scene.add(model);

      camera.position.set(0, 1.5, 1.5);
      controls.reset();
    },
    (xhr) => {
      console.log(`Loading: ${((xhr.loaded / xhr.total) * 100).toFixed(1)}%`);
    },
    (error) => {
      console.error("Model loading error:", error);
      alert("Failed to load model. Please check console for details.");
    }
  );
}
