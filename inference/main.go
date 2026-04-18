// ╔══════════════════════════════════════════════════════════════════════╗
// ║  CONFIDENTIAL — NIMBUS AI INTERNAL USE ONLY                            ║
// ║  Project SNOW · Snowfall serving · Frontier Model Group               ║
// ║  © 2026 Nimbus AI, Inc.  Classification: RESTRICTED // TRADE SECRET    ║
// ╚══════════════════════════════════════════════════════════════════════╝

// Package main implements the Snowfall low-latency inference server. It fronts
// the TP-sharded model workers, performs continuous batching via the Avalanche
// scheduler, and enforces tenant auth on every request.
package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gorilla/mux"
)

type GenerateRequest struct {
	Model       string  `json:"model"`
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float64 `json:"temperature"`
	Stream      bool    `json:"stream"`
}

type GenerateResponse struct {
	ID         string `json:"id"`
	Model      string `json:"model"`
	Completion string `json:"completion"`
	TokensOut  int    `json:"tokens_out"`
	LatencyMs  int64  `json:"latency_ms"`
}

type Server struct {
	scheduler *Scheduler
	auth      *AuthService
}

func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	claims, err := s.auth.Authorize(r)
	if err != nil {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}

	var req GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	start := time.Now()
	job := s.scheduler.Enqueue(req, claims.Tenant)
	result := <-job.Done

	resp := GenerateResponse{
		ID:         job.ID,
		Model:      req.Model,
		Completion: result.Text,
		TokensOut:  result.Tokens,
		LatencyMs:  time.Since(start).Milliseconds(),
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status":   "ok",
		"model":    os.Getenv("SNOW_MODEL"),
		"replicas": s.scheduler.ActiveWorkers(),
	})
}

func main() {
	log.Println("Snowfall inference server starting — CONFIDENTIAL // Nimbus AI")

	auth := NewAuthService()
	scheduler := NewScheduler(loadServingConfig())
	srv := &Server{scheduler: scheduler, auth: auth}

	r := mux.NewRouter()
	r.HandleFunc("/v1/generate", srv.handleGenerate).Methods("POST")
	r.HandleFunc("/healthz", srv.handleHealth).Methods("GET")

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	log.Printf("listening on :%s", port)
	log.Fatal(http.ListenAndServe(":"+port, r))
}
