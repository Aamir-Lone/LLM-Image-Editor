package handlers

import (
	"net/http"

	"github.com/gorilla/mux"
)

func RegisterHealthCheck(router *mux.Router) {
	router.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("Backend is healthy"))
	}).Methods("GET")
}
