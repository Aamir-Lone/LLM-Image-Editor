package handlers

import (
	"bytes"
	"io"
	"mime/multipart"
	"net/http"
)

func HandleImageProcessing(w http.ResponseWriter, r *http.Request) {
	// Parse form data
	err := r.ParseMultipartForm(10 << 20) // 10 MB max file size
	if err != nil {
		http.Error(w, "Unable to parse form", http.StatusBadRequest)
		return
	}

	file, _, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "Unable to retrieve file", http.StatusBadRequest)
		return
	}
	defer file.Close()

	task := r.FormValue("task")
	if task == "" {
		http.Error(w, "Task is required", http.StatusBadRequest)
		return
	}

	// Send the file to the Python service
	response, err := sendToPythonService(file, task)
	if err != nil {
		http.Error(w, "Error processing image", http.StatusInternalServerError)
		return
	}

	// Return the processed image
	w.Header().Set("Content-Type", "image/jpeg")
	io.Copy(w, response.Body)
}

func sendToPythonService(file io.Reader, task string) (*http.Response, error) {
	url := "http://127.0.0.1:3000/process-image"

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, err := writer.CreateFormFile("file", "uploaded.jpg")
	if err != nil {
		return nil, err
	}
	_, err = io.Copy(part, file)

	if err != nil {
		return nil, err
	}
	_ = writer.WriteField("task", task)
	writer.Close()

	request, err := http.NewRequest("POST", url, body)
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{}
	return client.Do(request)
}
