package agent

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"
)

// A2AAgent extends the base agent with Agent-to-Agent communication capabilities
type A2AAgent struct {
	*Agent                    // Embed base agent
	messageChannel chan AgentMessage
	collaborations map[string]*CollaborationParticipation
	peerAgents     map[string]*PeerAgent
	trainingJobs   map[string]*TrainingJobParticipation
	mutex          sync.RWMutex
	isListening    bool
	logger         *log.Logger
}

// PeerAgent represents another agent in the network
type PeerAgent struct {
	ID           string                 `json:"id"`
	Type         string                 `json:"type"`
	Capabilities map[string]interface{} `json:"capabilities"`
	Endpoint     string                 `json:"endpoint"`
	LastSeen     time.Time              `json:"last_seen"`
	Reputation   float64                `json:"reputation"`
}

// CollaborationParticipation tracks agent's involvement in collaborations
type CollaborationParticipation struct {
	ID            string                 `json:"id"`
	Role          string                 `json:"role"` // "coordinator", "trainer", "validator"
	Status        string                 `json:"status"`
	Participants  []string               `json:"participants"`
	TrainingJob   map[string]interface{} `json:"training_job"`
	Progress      float64                `json:"progress"`
	MyContribution float64               `json:"my_contribution"`
	StartTime     time.Time              `json:"start_time"`
	LastUpdate    time.Time              `json:"last_update"`
}

// TrainingJobParticipation tracks training job execution
type TrainingJobParticipation struct {
	JobID           string                 `json:"job_id"`
	CollaborationID string                 `json:"collaboration_id"`
	ModelType       string                 `json:"model_type"`
	MyRole          string                 `json:"my_role"`
	TrainingConfig  map[string]interface{} `json:"training_config"`
	DataSources     []interface{}          `json:"data_sources"`
	Progress        float64                `json:"progress"`
	Status          string                 `json:"status"`
	StartTime       time.Time              `json:"start_time"`
	Metrics         map[string]interface{} `json:"metrics"`
}

// AgentMessage for peer-to-peer communication (same as hub definition)
type AgentMessage struct {
	ID        string                 `json:"id"`
	From      string                 `json:"from"`
	To        string                 `json:"to"`
	Type      string                 `json:"type"`
	Content   map[string]interface{} `json:"content"`
	Timestamp time.Time              `json:"timestamp"`
	Priority  string                 `json:"priority"`
}

// NewA2AAgent creates an enhanced agent with A2A capabilities
func NewA2AAgent(baseAgent *Agent) *A2AAgent {
	return &A2AAgent{
		Agent:          baseAgent,
		messageChannel: make(chan AgentMessage, 100),
		collaborations: make(map[string]*CollaborationParticipation),
		peerAgents:     make(map[string]*PeerAgent),
		trainingJobs:   make(map[string]*TrainingJobParticipation),
		logger:         log.New(log.Writer(), "[A2A] ", log.LstdFlags),
	}
}

// ConnectToAgentNetwork registers with hub and starts A2A communication
func (a *A2AAgent) ConnectToAgentNetwork() error {
	// First, register with the hub using base agent functionality
	if err := a.Agent.Register(); err != nil {
		return fmt.Errorf("failed to register with hub: %w", err)
	}

	// Connect to agent communication hub
	capabilities := map[string]interface{}{
		"gpu_memory_mb":         8192, // From metrics
		"compute_power_score":   100,
		"supported_model_types": []string{"language_model", "vision_model", "multimodal_model"},
		"max_data_size_mb":      2000,
		"specialties":           []string{"training", "inference"},
		"location":              "auto-detect",
		"avg_latency_ms":        50,
	}

	agentID := fmt.Sprintf("agent_%x", a.PubKey[:8])
	connectReq := map[string]interface{}{
		"agent_id":     agentID,
		"type":         "trainer",
		"capabilities": capabilities,
		"metadata": map[string]interface{}{
			"version":     "1.0.0",
			"node_pubkey": fmt.Sprintf("%x", a.PubKey),
		},
	}

	if err := postJSON(a.HubBaseURL+"/api/v1/agents/connect", connectReq, nil); err != nil {
		a.logger.Printf("Warning: Could not connect to agent hub: %v", err)
		// Continue anyway - hub might not have A2A enabled yet
	}

	// Start listening for messages from hub
	go a.startMessageListener()

	a.logger.Printf("Agent %s connected to A2A network", agentID)
	return nil
}

// StartMessageListener polls for messages from the hub
func (a *A2AAgent) startMessageListener() {
	a.isListening = true
	agentID := fmt.Sprintf("agent_%x", a.PubKey[:8])
	
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for a.isListening {
		select {
		case <-ticker.C:
			a.pollForMessages(agentID)
		}
	}
}

// Poll for messages from the hub
func (a *A2AAgent) pollForMessages(agentID string) {
	url := fmt.Sprintf("%s/api/v1/agents/messages/%s", a.HubBaseURL, agentID)
	resp, err := http.Get(url)
	if err != nil {
		// Hub might not be available - that's ok
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return
	}

	var messageResp struct {
		Messages []AgentMessage `json:"messages"`
		Count    int            `json:"count"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&messageResp); err != nil {
		return
	}

	// Process each message
	for _, message := range messageResp.Messages {
		a.handleIncomingMessage(message)
	}
}

// Handle incoming messages from other agents or hub
func (a *A2AAgent) handleIncomingMessage(message AgentMessage) {
	a.logger.Printf("Received %s message from %s", message.Type, message.From)

	switch message.Type {
	case "training_invitation":
		a.handleTrainingInvitation(message)
	case "collaboration_invite":
		a.handleCollaborationInvite(message)
	case "progress_request":
		a.handleProgressRequest(message)
	case "training_completed":
		a.handleTrainingCompleted(message)
	case "training_timeout":
		a.handleTrainingTimeout(message)
	case "peer_discovery":
		a.handlePeerDiscovery(message)
	case "coordination_request":
		a.handleCoordinationRequest(message)
	default:
		a.logger.Printf("Unknown message type: %s", message.Type)
	}
}

// Handle training invitation from hub
func (a *A2AAgent) handleTrainingInvitation(message AgentMessage) {
	content := message.Content
	collaborationID, _ := content["collaboration_id"].(string)
	trainingJobID, _ := content["training_job_id"].(string)
	modelType, _ := content["model_type"].(string)
	estimatedReward, _ := content["estimated_reward"].(float64)

	a.logger.Printf("Received training invitation for %s (reward: $%.2f)", modelType, estimatedReward)

	// Evaluate if we want to participate
	if a.shouldAcceptTraining(content) {
		a.acceptTrainingInvitation(collaborationID, trainingJobID, content)
	} else {
		a.declineTrainingInvitation(collaborationID, trainingJobID, "insufficient_resources")
	}
}

// Decide whether to accept training based on our capabilities and load
func (a *A2AAgent) shouldAcceptTraining(content map[string]interface{}) bool {
	modelType, _ := content["model_type"].(string)
	estimatedTime, _ := content["estimated_time"].(float64)
	
	// Simple decision logic - accept if we support the model type and have capacity
	supportedTypes := []string{"language_model", "vision_model", "multimodal_model"}
	for _, supported := range supportedTypes {
		if supported == modelType {
			// Check if we're not too busy (simple heuristic)
			a.mutex.RLock()
			activeJobs := len(a.trainingJobs)
			a.mutex.RUnlock()
			
			if activeJobs < 2 && estimatedTime < 120 { // Less than 2 hours
				return true
			}
		}
	}
	return false
}

// Accept training invitation and join collaboration
func (a *A2AAgent) acceptTrainingInvitation(collaborationID, trainingJobID string, content map[string]interface{}) {
	agentID := fmt.Sprintf("agent_%x", a.PubKey[:8])

	// Join the collaboration
	joinReq := map[string]interface{}{
		"agent_id": agentID,
	}

	joinURL := fmt.Sprintf("%s/api/v1/collaborations/%s/join", a.HubBaseURL, collaborationID)
	if err := postJSON(joinURL, joinReq, nil); err != nil {
		a.logger.Printf("Failed to join collaboration: %v", err)
		return
	}

	// Create local tracking
	collaboration := &CollaborationParticipation{
		ID:            collaborationID,
		Role:          "trainer",
		Status:        "accepted",
		Participants:  []string{agentID},
		TrainingJob:   content,
		Progress:      0.0,
		MyContribution: 0.0,
		StartTime:     time.Now(),
		LastUpdate:    time.Now(),
	}

	trainingJob := &TrainingJobParticipation{
		JobID:           trainingJobID,
		CollaborationID: collaborationID,
		ModelType:       content["model_type"].(string),
		MyRole:          "trainer",
		TrainingConfig:  content["training_config"].(map[string]interface{}),
		DataSources:     content["data_sources"].([]interface{}),
		Progress:        0.0,
		Status:          "starting",
		StartTime:       time.Now(),
		Metrics:         make(map[string]interface{}),
	}

	a.mutex.Lock()
	a.collaborations[collaborationID] = collaboration
	a.trainingJobs[trainingJobID] = trainingJob
	a.mutex.Unlock()

	// Start training in a separate goroutine
	go a.executeTrainingJob(trainingJob)

	a.logger.Printf("Accepted training invitation for collaboration %s", collaborationID)
}

// Decline training invitation
func (a *A2AAgent) declineTrainingInvitation(collaborationID, trainingJobID, reason string) {
	agentID := fmt.Sprintf("agent_%x", a.PubKey[:8])
	
	// Send decline message back to hub
	declineMessage := map[string]interface{}{
		"from":             agentID,
		"to":               "training_coordinator",
		"type":             "training_declined",
		"content": map[string]interface{}{
			"collaboration_id": collaborationID,
			"training_job_id":  trainingJobID,
			"reason":          reason,
		},
		"priority": "normal",
	}

	postJSON(a.HubBaseURL+"/api/v1/agents/message", declineMessage, nil)
	a.logger.Printf("Declined training invitation: %s", reason)
}

// Execute training job as part of collaboration
func (a *A2AAgent) executeTrainingJob(job *TrainingJobParticipation) {
	a.logger.Printf("Starting training execution for job %s", job.JobID)
	
	startTime := time.Now()
	
	// Update status
	job.Status = "training"
	job.Progress = 0.0

	// Simulate training process with progress updates
	steps := 10
	for i := 0; i < steps; i++ {
		if job.Status == "cancelled" {
			break
		}

		// Simulate training step
		time.Sleep(10 * time.Second) // Simulate training time
		
		progress := float64(i+1) / float64(steps)
		job.Progress = progress

		// Send progress update to hub
		a.sendProgressUpdate(job)

		a.logger.Printf("Training progress: %.1f%%", progress*100)
	}

	// Complete training
	duration := time.Since(startTime)
	job.Status = "completed"
	job.Progress = 1.0
	job.Metrics = map[string]interface{}{
		"duration_seconds":   duration.Seconds(),
		"final_accuracy":     0.85 + (0.1 * float64(len(job.JobID)%10)/10), // Simulate varying accuracy
		"training_loss":      0.15,
		"validation_loss":    0.12,
		"gpu_utilization":    95.0,
		"memory_usage_gb":    6.5,
	}

	// Send final progress update
	a.sendProgressUpdate(job)
	
	// Send completion notification
	a.sendTrainingCompletion(job)

	a.logger.Printf("Training job %s completed successfully", job.JobID)
}

// Send progress update to hub
func (a *A2AAgent) sendProgressUpdate(job *TrainingJobParticipation) {
	agentID := fmt.Sprintf("agent_%x", a.PubKey[:8])
	
	updateReq := map[string]interface{}{
		"agent_id": agentID,
		"progress": job.Progress,
		"update_message": fmt.Sprintf("Training step completed, %s", job.Status),
	}

	updateURL := fmt.Sprintf("%s/api/v1/collaborations/%s/progress", a.HubBaseURL, job.CollaborationID)
	if err := postJSON(updateURL, updateReq, nil); err != nil {
		a.logger.Printf("Failed to send progress update: %v", err)
	}
}

// Send training completion notification
func (a *A2AAgent) sendTrainingCompletion(job *TrainingJobParticipation) {
	agentID := fmt.Sprintf("agent_%x", a.PubKey[:8])
	
	completionMessage := map[string]interface{}{
		"from": agentID,
		"to":   "training_coordinator",
		"type": "training_completed",
		"content": map[string]interface{}{
			"collaboration_id": job.CollaborationID,
			"training_job_id":  job.JobID,
			"final_metrics":    job.Metrics,
			"completion_time":  time.Now().Unix(),
		},
		"priority": "high",
	}

	postJSON(a.HubBaseURL+"/api/v1/agents/message", completionMessage, nil)
}

// Handle progress request from hub
func (a *A2AAgent) handleProgressRequest(message AgentMessage) {
	collaborationID, _ := message.Content["collaboration_id"].(string)
	
	a.mutex.RLock()
	collaboration, exists := a.collaborations[collaborationID]
	a.mutex.RUnlock()

	if !exists {
		return
	}

	// Send current progress
	agentID := fmt.Sprintf("agent_%x", a.PubKey[:8])
	progressMessage := map[string]interface{}{
		"from": agentID,
		"to":   message.From,
		"type": "progress_update",
		"content": map[string]interface{}{
			"collaboration_id": collaborationID,
			"progress":         collaboration.Progress,
			"status":          collaboration.Status,
			"last_update":     collaboration.LastUpdate.Unix(),
		},
		"priority": "normal",
	}

	postJSON(a.HubBaseURL+"/api/v1/agents/message", progressMessage, nil)
}

// Handle collaboration invitation
func (a *A2AAgent) handleCollaborationInvite(message AgentMessage) {
	content := message.Content
	collaborationID, _ := content["collaboration_id"].(string)
	budget, _ := content["budget"].(float64)
	
	a.logger.Printf("Received collaboration invite for %s (budget: $%.2f)", collaborationID, budget)
	
	// Simple acceptance logic - accept if budget is reasonable
	if budget > 5.0 {
		agentID := fmt.Sprintf("agent_%x", a.PubKey[:8])
		joinReq := map[string]interface{}{
			"agent_id": agentID,
		}
		
		joinURL := fmt.Sprintf("%s/api/v1/collaborations/%s/join", a.HubBaseURL, collaborationID)
		postJSON(joinURL, joinReq, nil)
		
		a.logger.Printf("Accepted collaboration invitation")
	}
}

// Handle training completion notification
func (a *A2AAgent) handleTrainingCompleted(message AgentMessage) {
	content := message.Content
	collaborationID, _ := content["collaboration_id"].(string)
	rewardAmount, _ := content["reward_amount"].(map[string]interface{})
	
	agentID := fmt.Sprintf("agent_%x", a.PubKey[:8])
	if reward, ok := rewardAmount[agentID].(float64); ok {
		a.logger.Printf("Training completed! Earned $%.2f", reward)
	}

	// Clean up local state
	a.mutex.Lock()
	delete(a.collaborations, collaborationID)
	a.mutex.Unlock()
}

// Handle training timeout
func (a *A2AAgent) handleTrainingTimeout(message AgentMessage) {
	content := message.Content
	collaborationID, _ := content["collaboration_id"].(string)
	reason, _ := content["reason"].(string)
	
	a.logger.Printf("Training timed out: %s", reason)
	
	// Cancel any ongoing training
	a.mutex.Lock()
	if collaboration, exists := a.collaborations[collaborationID]; exists {
		collaboration.Status = "timeout"
		for _, job := range a.trainingJobs {
			if job.CollaborationID == collaborationID {
				job.Status = "cancelled"
			}
		}
	}
	a.mutex.Unlock()
}

// Handle peer discovery messages
func (a *A2AAgent) handlePeerDiscovery(message AgentMessage) {
	// Extract peer information and add to known peers
	content := message.Content
	peerID, _ := content["peer_id"].(string)
	peerType, _ := content["peer_type"].(string)
	capabilities, _ := content["capabilities"].(map[string]interface{})
	
	peer := &PeerAgent{
		ID:           peerID,
		Type:         peerType,
		Capabilities: capabilities,
		LastSeen:     time.Now(),
		Reputation:   1.0,
	}

	a.mutex.Lock()
	a.peerAgents[peerID] = peer
	a.mutex.Unlock()

	a.logger.Printf("Discovered peer agent: %s (%s)", peerID, peerType)
}

// Handle coordination requests from other agents
func (a *A2AAgent) handleCoordinationRequest(message AgentMessage) {
	content := message.Content
	objective, _ := content["objective"].(string)
	budget, _ := content["budget"].(float64)
	
	a.logger.Printf("Received coordination request: %s (budget: $%.2f)", objective, budget)
	
	// Evaluate and respond
	if a.canParticipateInCoordination(content) {
		a.acceptCoordinationRequest(message)
	} else {
		a.declineCoordinationRequest(message, "insufficient_capacity")
	}
}

// Evaluate coordination participation
func (a *A2AAgent) canParticipateInCoordination(content map[string]interface{}) bool {
	a.mutex.RLock()
	activeCollabs := len(a.collaborations)
	a.mutex.RUnlock()
	
	return activeCollabs < 3 // Can handle up to 3 simultaneous collaborations
}

// Accept coordination request
func (a *A2AAgent) acceptCoordinationRequest(message AgentMessage) {
	agentID := fmt.Sprintf("agent_%x", a.PubKey[:8])
	
	responseMessage := map[string]interface{}{
		"from": agentID,
		"to":   message.From,
		"type": "coordination_accepted",
		"content": map[string]interface{}{
			"original_request": message.Content,
			"available_capabilities": map[string]interface{}{
				"gpu_memory":    8192,
				"compute_power": 100,
				"specialties":   []string{"training", "validation"},
			},
		},
		"priority": "high",
	}

	postJSON(a.HubBaseURL+"/api/v1/agents/message", responseMessage, nil)
	a.logger.Printf("Accepted coordination request from %s", message.From)
}

// Decline coordination request
func (a *A2AAgent) declineCoordinationRequest(message AgentMessage, reason string) {
	agentID := fmt.Sprintf("agent_%x", a.PubKey[:8])
	
	responseMessage := map[string]interface{}{
		"from": agentID,
		"to":   message.From,
		"type": "coordination_declined",
		"content": map[string]interface{}{
			"reason": reason,
		},
		"priority": "normal",
	}

	postJSON(a.HubBaseURL+"/api/v1/agents/message", responseMessage, nil)
	a.logger.Printf("Declined coordination request: %s", reason)
}

// Send heartbeat to both hub and agent network
func (a *A2AAgent) EnhancedHeartbeat() error {
	// Send regular heartbeat to hub
	if err := a.Agent.HeartbeatOnce(); err != nil {
		return err
	}

	// Send agent heartbeat to agent network
	agentID := fmt.Sprintf("agent_%x", a.PubKey[:8])
	agentHeartbeat := map[string]interface{}{
		"agent_id": agentID,
		"status":   "active",
		"current_load": a.calculateCurrentLoad(),
	}

	postJSON(a.HubBaseURL+"/api/v1/agents/heartbeat", agentHeartbeat, nil)
	return nil
}

// Calculate current load for capacity planning
func (a *A2AAgent) calculateCurrentLoad() float64 {
	a.mutex.RLock()
	activeJobs := len(a.trainingJobs)
	activeCollabs := len(a.collaborations)
	a.mutex.RUnlock()

	// Simple load calculation
	load := float64(activeJobs)*0.3 + float64(activeCollabs)*0.2
	if load > 1.0 {
		load = 1.0
	}
	return load
}

// Stop A2A communication
func (a *A2AAgent) StopA2A() {
	a.isListening = false
	
	// Send disconnect message
	agentID := fmt.Sprintf("agent_%x", a.PubKey[:8])
	disconnectReq := map[string]interface{}{
		"agent_id": agentID,
		"reason":   "normal_shutdown",
	}

	postJSON(a.HubBaseURL+"/api/v1/agents/disconnect", disconnectReq, nil)
	a.logger.Printf("A2A agent disconnected")
}

// Get status of A2A agent
func (a *A2AAgent) GetA2AStatus() map[string]interface{} {
	a.mutex.RLock()
	defer a.mutex.RUnlock()

	return map[string]interface{}{
		"agent_id":           fmt.Sprintf("agent_%x", a.PubKey[:8]),
		"is_listening":       a.isListening,
		"active_collaborations": len(a.collaborations),
		"active_training_jobs": len(a.trainingJobs),
		"known_peers":        len(a.peerAgents),
		"current_load":       a.calculateCurrentLoad(),
		"status":            "active",
	}
}