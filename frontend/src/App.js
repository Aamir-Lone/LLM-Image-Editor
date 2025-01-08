document.getElementById("uploadForm").addEventListener("submit", async (event) => {
    event.preventDefault();

    const fileInput = document.getElementById("fileInput");
    const taskSelect = document.getElementById("taskSelect");
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("task", taskSelect.value);

    const response = await fetch("http://127.0.0.1:3000/process-image", {
        method: "POST",
        body: formData,
    });

    if (response.ok) {
        const blob = await response.blob();
        document.getElementById("resultImage").src = URL.createObjectURL(blob);
    } else {
        alert("Error processing image");
    }
});
