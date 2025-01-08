package main

import (
	"log"
	"net/http"

	"LLM_image_editor/backend/handlers"

	"github.com/gorilla/mux"
)

func main() {
	r := mux.NewRouter()
	r.HandleFunc("/process-image", handlers.HandleImageProcessing).Methods("POST")

	log.Println("Server running on http://localhost:8080")
	log.Fatal(http.ListenAndServe(":8080", r))
}
