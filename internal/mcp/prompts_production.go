// File: node-agent/internal/mcp/prompts_production.go
package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"time"
)

// GetPrompt retrieves a specific prompt by name with optional arguments
func (c *ProductionMCPClient) GetPrompt(name string, arguments map[string]interface{}) (*MCPPromptResult, error) {
	start := time.Now()
	defer func() {
		c.metrics.TotalRequests++
		responseTime := time.Since(start)
		if c.metrics.AverageResponseTime == 0 {
			c.metrics.AverageResponseTime = responseTime
		} else {
			alpha := 0.1
			c.metrics.AverageResponseTime = time.Duration(float64(c.metrics.AverageResponseTime)*(1-alpha) + float64(responseTime)*alpha)
		}
	}()
	
	if !c.IsConnected() {
		return nil, fmt.Errorf("client not connected")
	}
	
	// Verify prompt exists
	c.dataMutex.RLock()
	prompt, exists := c.prompts[name]
	c.dataMutex.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("prompt '%s' not found", name)
	}
	
	// Validate arguments against prompt requirements
	if err := c.validatePromptArguments(prompt, arguments); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}
	
	// Create prompt get message
	getMsg := &MCPMessage{
		JSONRPCVersion: "2.0",
		ID:             c.nextRequestID(),
		Method:         "prompts/get",
		Params: map[string]interface{}{
			"name":      name,
			"arguments": arguments,
		},
	}
	
	c.logger.Printf("Getting prompt '%s' with arguments: %v", name, arguments)
	
	// Send request and wait for response
	response, err := c.sendRequestWithResponse(getMsg)
	if err != nil {
		c.metrics.FailedRequests++
		return nil, fmt.Errorf("prompt get failed: %w", err)
	}
	
	if response.Error != nil {
		c.metrics.FailedRequests++
		return nil, fmt.Errorf("prompt get error: %s", response.Error.Message)
	}
	
	// Parse prompt result
	if result, ok := response.Result.(map[string]interface{}); ok {
		resultBytes, _ := json.Marshal(result)
		var promptResult MCPPromptResult
		if err := json.Unmarshal(resultBytes, &promptResult); err != nil {
			c.metrics.FailedRequests++
			return nil, fmt.Errorf("failed to parse prompt result: %w", err)
		}
		
		c.metrics.SuccessfulRequests++
		c.metrics.PromptsRetrieved++
		c.logger.Printf("Successfully retrieved prompt '%s' (%d messages)", name, len(promptResult.Messages))
		return &promptResult, nil
	}
	
	c.metrics.FailedRequests++
	return nil, fmt.Errorf("invalid prompt result format")
}

// GetPromptAsync retrieves a prompt asynchronously
func (c *ProductionMCPClient) GetPromptAsync(name string, arguments map[string]interface{}) (<-chan *MCPPromptResult, <-chan error) {
	resultChan := make(chan *MCPPromptResult, 1)
	errorChan := make(chan error, 1)
	
	go func() {
		defer close(resultChan)
		defer close(errorChan)
		
		result, err := c.GetPrompt(name, arguments)
		if err != nil {
			errorChan <- err
			return
		}
		
		resultChan <- result
	}()
	
	return resultChan, errorChan
}

// GetPromptWithTimeout retrieves a prompt with custom timeout
func (c *ProductionMCPClient) GetPromptWithTimeout(name string, arguments map[string]interface{}, timeout time.Duration) (*MCPPromptResult, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	
	resultChan, errorChan := c.GetPromptAsync(name, arguments)
	
	select {
	case result := <-resultChan:
		return result, nil
	case err := <-errorChan:
		return nil, err
	case <-ctx.Done():
		return nil, fmt.Errorf("prompt get timeout after %v", timeout)
	}
}

// GetPromptWithContext retrieves a prompt with the given context
func (c *ProductionMCPClient) GetPromptWithContext(ctx context.Context, name string, arguments map[string]interface{}) (*MCPPromptResult, error) {
	type result struct {
		promptResult *MCPPromptResult
		err          error
	}
	
	resultChan := make(chan result, 1)
	
	go func() {
		promptResult, err := c.GetPrompt(name, arguments)
		resultChan <- result{promptResult: promptResult, err: err}
	}()
	
	select {
	case res := <-resultChan:
		return res.promptResult, res.err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// ListPrompts returns all available prompts
func (c *ProductionMCPClient) ListPrompts() ([]*MCPPrompt, error) {
	c.dataMutex.RLock()
	defer c.dataMutex.RUnlock()
	
	prompts := make([]*MCPPrompt, 0, len(c.prompts))
	for _, prompt := range c.prompts {
		prompts = append(prompts, prompt)
	}
	
	return prompts, nil
}

// FindPromptsByCategory filters prompts by a category or tag
func (c *ProductionMCPClient) FindPromptsByCategory(category string) ([]*MCPPrompt, error) {
	allPrompts, err := c.ListPrompts()
	if err != nil {
		return nil, err
	}
	
	var filteredPrompts []*MCPPrompt
	for _, prompt := range allPrompts {
		// Check if prompt description contains category (enhanced filtering)
		if c.promptMatchesCategory(prompt, category) {
			filteredPrompts = append(filteredPrompts, prompt)
		}
	}
	
	return filteredPrompts, nil
}

// promptMatchesCategory checks if a prompt matches a given category
func (c *ProductionMCPClient) promptMatchesCategory(prompt *MCPPrompt, category string) bool {
	// Check name
	if contains(prompt.Name, category) {
		return true
	}
	
	// Check description
	if contains(prompt.Description, category) {
		return true
	}
	
	// Check arguments for category hints
	for _, arg := range prompt.Arguments {
		if contains(arg.Name, category) || contains(arg.Description, category) {
			return true
		}
	}
	
	return false
}

// validatePromptArguments validates arguments against prompt requirements
func (c *ProductionMCPClient) validatePromptArguments(prompt *MCPPrompt, arguments map[string]interface{}) error {
	if prompt.Arguments == nil || len(prompt.Arguments) == 0 {
		return nil // No arguments required
	}
	
	// Check required arguments
	for _, arg := range prompt.Arguments {
		if arg.Required {
			if arguments == nil {
				return fmt.Errorf("required argument '%s' missing for prompt '%s'", arg.Name, prompt.Name)
			}
			if _, exists := arguments[arg.Name]; !exists {
				return fmt.Errorf("required argument '%s' missing for prompt '%s'", arg.Name, prompt.Name)
			}
		}
	}
	
	// Validate argument types and constraints (enhanced validation)
	if arguments != nil {
		for argName, argValue := range arguments {
			if err := c.validatePromptArgumentValue(prompt, argName, argValue); err != nil {
				return fmt.Errorf("argument '%s' validation failed: %w", argName, err)
			}
		}
	}
	
	return nil
}

// validatePromptArgumentValue validates individual prompt argument values
func (c *ProductionMCPClient) validatePromptArgumentValue(prompt *MCPPrompt, argName string, value interface{}) error {
	// Find the argument definition
	var argDef *MCPPromptArgument
	for _, arg := range prompt.Arguments {
		if arg.Name == argName {
			argDef = &arg
			break
		}
	}
	
	if argDef == nil {
		// Unknown argument - could be optional context
		return nil
	}
	
	// Basic type validation based on argument description
	if contains(argDef.Description, "string") {
		if _, ok := value.(string); !ok {
			return fmt.Errorf("expected string value for argument '%s'", argName)
		}
	} else if contains(argDef.Description, "number") || contains(argDef.Description, "integer") {
		switch value.(type) {
		case int, int32, int64, float32, float64:
			// Valid number types
		default:
			return fmt.Errorf("expected number value for argument '%s'", argName)
		}
	} else if contains(argDef.Description, "boolean") {
		if _, ok := value.(bool); !ok {
			return fmt.Errorf("expected boolean value for argument '%s'", argName)
		}
	} else if contains(argDef.Description, "array") || contains(argDef.Description, "list") {
		if _, ok := value.([]interface{}); !ok {
			return fmt.Errorf("expected array value for argument '%s'", argName)
		}
	}
	
	return nil
}

// Convenience methods for common prompt patterns

// GetSystemPrompt retrieves a system-level prompt
func (c *ProductionMCPClient) GetSystemPrompt(task string, context map[string]interface{}) (*MCPPromptResult, error) {
	if task == "" {
		return nil, fmt.Errorf("system task cannot be empty")
	}
	
	arguments := map[string]interface{}{
		"task": task,
	}
	
	if context != nil {
		arguments["context"] = context
	}
	
	return c.GetPrompt("system", arguments)
}

// GetConversationPrompt retrieves a conversation prompt for chat interfaces
func (c *ProductionMCPClient) GetConversationPrompt(userMessage string, conversationHistory []map[string]interface{}) (*MCPPromptResult, error) {
	if userMessage == "" {
		return nil, fmt.Errorf("user message cannot be empty")
	}
	
	arguments := map[string]interface{}{
		"user_message": userMessage,
	}
	
	if conversationHistory != nil && len(conversationHistory) > 0 {
		arguments["conversation_history"] = conversationHistory
	}
	
	return c.GetPrompt("conversation", arguments)
}

// GetTaskPrompt retrieves a task-specific prompt
func (c *ProductionMCPClient) GetTaskPrompt(taskType string, parameters map[string]interface{}) (*MCPPromptResult, error) {
	if taskType == "" {
		return nil, fmt.Errorf("task type cannot be empty")
	}
	
	arguments := map[string]interface{}{
		"task_type": taskType,
	}
	
	if parameters != nil {
		arguments["parameters"] = parameters
	}
	
	return c.GetPrompt("task", arguments)
}

// GetRolePrompt retrieves a role-based prompt for different agent personas
func (c *ProductionMCPClient) GetRolePrompt(role string, instructions string, context map[string]interface{}) (*MCPPromptResult, error) {
	if role == "" {
		return nil, fmt.Errorf("role cannot be empty")
	}
	
	arguments := map[string]interface{}{
		"role": role,
	}
	
	if instructions != "" {
		arguments["instructions"] = instructions
	}
	
	if context != nil {
		arguments["context"] = context
	}
	
	return c.GetPrompt("role", arguments)
}

// GetAnalysisPrompt retrieves a prompt for data analysis tasks
func (c *ProductionMCPClient) GetAnalysisPrompt(dataType string, analysisGoal string, data interface{}) (*MCPPromptResult, error) {
	if dataType == "" || analysisGoal == "" {
		return nil, fmt.Errorf("data type and analysis goal cannot be empty")
	}
	
	arguments := map[string]interface{}{
		"data_type":     dataType,
		"analysis_goal": analysisGoal,
	}
	
	if data != nil {
		arguments["data"] = data
	}
	
	return c.GetPrompt("analysis", arguments)
}

// GetCreativePrompt retrieves a prompt for creative tasks
func (c *ProductionMCPClient) GetCreativePrompt(creativeType string, style string, constraints map[string]interface{}) (*MCPPromptResult, error) {
	if creativeType == "" {
		return nil, fmt.Errorf("creative type cannot be empty")
	}
	
	arguments := map[string]interface{}{
		"creative_type": creativeType,
	}
	
	if style != "" {
		arguments["style"] = style
	}
	
	if constraints != nil {
		arguments["constraints"] = constraints
	}
	
	return c.GetPrompt("creative", arguments)
}

// GetEnterprisePrompt retrieves enterprise-specific prompts with compliance
func (c *ProductionMCPClient) GetEnterprisePrompt(promptType string, department string, complianceLevel string, context map[string]interface{}) (*MCPPromptResult, error) {
	if promptType == "" || department == "" {
		return nil, fmt.Errorf("prompt type and department cannot be empty")
	}
	
	arguments := map[string]interface{}{
		"prompt_type": promptType,
		"department":  department,
	}
	
	if complianceLevel != "" {
		arguments["compliance_level"] = complianceLevel
	}
	
	if context != nil {
		arguments["context"] = context
	}
	
	promptName := fmt.Sprintf("enterprise_%s", promptType)
	return c.GetPrompt(promptName, arguments)
}

// Batch prompt operations

// BatchGetPrompts retrieves multiple prompts in parallel
func (c *ProductionMCPClient) BatchGetPrompts(requests []PromptRequest) (map[string]*MCPPromptResult, map[string]error) {
	if len(requests) == 0 {
		return make(map[string]*MCPPromptResult), make(map[string]error)
	}
	
	results := make(map[string]*MCPPromptResult)
	errors := make(map[string]error)
	
	type result struct {
		name   string
		prompt *MCPPromptResult
		err    error
	}
	
	resultChan := make(chan result, len(requests))
	
	// Start all requests in parallel
	for _, req := range requests {
		go func(r PromptRequest) {
			prompt, err := c.GetPrompt(r.Name, r.Arguments)
			resultChan <- result{name: r.Name, prompt: prompt, err: err}
		}(req)
	}
	
	// Collect results
	for i := 0; i < len(requests); i++ {
		res := <-resultChan
		if res.err != nil {
			errors[res.name] = res.err
		} else {
			results[res.name] = res.prompt
		}
	}
	
	return results, errors
}

// PromptRequest represents a prompt request for batch operations
type PromptRequest struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments,omitempty"`
}

// Advanced prompt chaining and composition

// ExecutePromptChain executes a sequence of prompts with context passing
func (c *ProductionMCPClient) ExecutePromptChain(chain []PromptChainStep) (*PromptChainResult, error) {
	if len(chain) == 0 {
		return nil, fmt.Errorf("prompt chain cannot be empty")
	}
	
	result := &PromptChainResult{
		Steps:   make([]PromptStepResult, len(chain)),
		Context: make(map[string]interface{}),
	}
	
	for i, step := range chain {
		// Merge previous results into context
		arguments := make(map[string]interface{})
		if step.Arguments != nil {
			for k, v := range step.Arguments {
				arguments[k] = v
			}
		}
		
		// Add context from previous steps
		if i > 0 && step.UseContext {
			for k, v := range result.Context {
				if _, exists := arguments[k]; !exists {
					arguments[k] = v
				}
			}
		}
		
		// Execute prompt
		promptResult, err := c.GetPrompt(step.PromptName, arguments)
		
		stepResult := PromptStepResult{
			StepName:    step.Name,
			PromptName:  step.PromptName,
			Result:      promptResult,
			Error:       err,
		}
		
		result.Steps[i] = stepResult
		
		// Stop on error if specified
		if err != nil {
			if step.StopOnError {
				result.Success = false
				result.Error = fmt.Sprintf("Step '%s' failed: %v", step.Name, err)
				return result, err
			}
		} else {
			// Add result to context for next steps
			if step.OutputVariable != "" && promptResult != nil {
				result.Context[step.OutputVariable] = promptResult.GetCombinedText()
			}
		}
	}
	
	result.Success = true
	return result, nil
}

// PromptChainStep represents a step in a prompt chain
type PromptChainStep struct {
	Name           string                 `json:"name"`
	PromptName     string                 `json:"prompt_name"`
	Arguments      map[string]interface{} `json:"arguments,omitempty"`
	UseContext     bool                   `json:"use_context,omitempty"`
	OutputVariable string                 `json:"output_variable,omitempty"`
	StopOnError    bool                   `json:"stop_on_error,omitempty"`
}

// PromptChainResult represents the result of executing a prompt chain
type PromptChainResult struct {
	Success bool                         `json:"success"`
	Steps   []PromptStepResult           `json:"steps"`
	Context map[string]interface{}       `json:"context"`
	Error   string                       `json:"error,omitempty"`
}

// PromptStepResult represents the result of a single step in a prompt chain
type PromptStepResult struct {
	StepName   string            `json:"step_name"`
	PromptName string            `json:"prompt_name"`
	Result     *MCPPromptResult  `json:"result,omitempty"`
	Error      error             `json:"error,omitempty"`
}