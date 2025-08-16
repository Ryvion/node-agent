// File: node-agent/internal/mcp/tools_production.go
package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// CallTool executes a tool with the given arguments
func (c *ProductionMCPClient) CallTool(toolName string, arguments interface{}) (*MCPToolResult, error) {
	start := time.Now()
	defer func() {
		c.metrics.TotalRequests++
		responseTime := time.Since(start)
		// Update average response time
		if c.metrics.AverageResponseTime == 0 {
			c.metrics.AverageResponseTime = responseTime
		} else {
			// Exponential moving average
			alpha := 0.1
			c.metrics.AverageResponseTime = time.Duration(float64(c.metrics.AverageResponseTime)*(1-alpha) + float64(responseTime)*alpha)
		}
	}()
	
	if !c.IsConnected() {
		return nil, fmt.Errorf("client not connected")
	}
	
	// Verify tool exists
	c.dataMutex.RLock()
	tool, exists := c.tools[toolName]
	c.dataMutex.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("tool '%s' not found", toolName)
	}
	
	// Validate arguments against schema
	if err := c.validateToolArguments(tool, arguments); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}
	
	// Create tool call message
	callMsg := &MCPMessage{
		JSONRPCVersion: "2.0",
		ID:             c.nextRequestID(),
		Method:         "tools/call",
		Params: map[string]interface{}{
			"name":      toolName,
			"arguments": arguments,
		},
	}
	
	c.logger.Printf("Calling tool '%s' with arguments: %v", toolName, arguments)
	
	// Send request and wait for response
	response, err := c.sendRequestWithResponse(callMsg)
	if err != nil {
		c.metrics.FailedRequests++
		return nil, fmt.Errorf("tool call failed: %w", err)
	}
	
	if response.Error != nil {
		c.metrics.FailedRequests++
		return &MCPToolResult{
			IsError: true,
			Content: []MCPContent{
				{
					Type: "text",
					Text: fmt.Sprintf("Tool error: %s", response.Error.Message),
				},
			},
		}, nil
	}
	
	// Parse tool result
	if result, ok := response.Result.(map[string]interface{}); ok {
		resultBytes, _ := json.Marshal(result)
		var toolResult MCPToolResult
		if err := json.Unmarshal(resultBytes, &toolResult); err != nil {
			c.metrics.FailedRequests++
			return nil, fmt.Errorf("failed to parse tool result: %w", err)
		}
		
		c.metrics.SuccessfulRequests++
		c.metrics.ToolCallsExecuted++
		c.logger.Printf("Tool '%s' completed successfully", toolName)
		return &toolResult, nil
	}
	
	c.metrics.FailedRequests++
	return nil, fmt.Errorf("invalid tool result format")
}

// CallToolAsync executes a tool asynchronously
func (c *ProductionMCPClient) CallToolAsync(toolName string, arguments interface{}) (<-chan *MCPToolResult, <-chan error) {
	resultChan := make(chan *MCPToolResult, 1)
	errorChan := make(chan error, 1)
	
	go func() {
		defer close(resultChan)
		defer close(errorChan)
		
		result, err := c.CallTool(toolName, arguments)
		if err != nil {
			errorChan <- err
			return
		}
		
		resultChan <- result
	}()
	
	return resultChan, errorChan
}

// CallToolWithTimeout executes a tool with a custom timeout
func (c *ProductionMCPClient) CallToolWithTimeout(toolName string, arguments interface{}, timeout time.Duration) (*MCPToolResult, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	
	resultChan, errorChan := c.CallToolAsync(toolName, arguments)
	
	select {
	case result := <-resultChan:
		return result, nil
	case err := <-errorChan:
		return nil, err
	case <-ctx.Done():
		return nil, fmt.Errorf("tool call timeout after %v", timeout)
	}
}

// CallToolWithContext executes a tool with the given context
func (c *ProductionMCPClient) CallToolWithContext(ctx context.Context, toolName string, arguments interface{}) (*MCPToolResult, error) {
	type result struct {
		toolResult *MCPToolResult
		err        error
	}
	
	resultChan := make(chan result, 1)
	
	go func() {
		toolResult, err := c.CallTool(toolName, arguments)
		resultChan <- result{toolResult: toolResult, err: err}
	}()
	
	select {
	case res := <-resultChan:
		return res.toolResult, res.err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// validateToolArguments validates arguments against tool schema
func (c *ProductionMCPClient) validateToolArguments(tool *MCPTool, arguments interface{}) error {
	// Enhanced validation with detailed error messages
	if tool.InputSchema.Required != nil && len(tool.InputSchema.Required) > 0 {
		if args, ok := arguments.(map[string]interface{}); ok {
			for _, required := range tool.InputSchema.Required {
				if _, exists := args[required]; !exists {
					return fmt.Errorf("required argument '%s' missing for tool '%s'", required, tool.Name)
				}
			}
			
			// Type validation for known properties
			if tool.InputSchema.Properties != nil {
				for argName, argValue := range args {
					if propSchema, exists := tool.InputSchema.Properties[argName]; exists {
						if err := c.validateArgumentType(argName, argValue, propSchema); err != nil {
							return fmt.Errorf("argument '%s' validation failed: %w", argName, err)
						}
					}
				}
			}
		} else {
			return fmt.Errorf("arguments must be an object for tool '%s'", tool.Name)
		}
	}
	
	return nil
}

// validateArgumentType validates individual argument types
func (c *ProductionMCPClient) validateArgumentType(argName string, value interface{}, schema interface{}) error {
	if schemaMap, ok := schema.(map[string]interface{}); ok {
		if expectedType, ok := schemaMap["type"].(string); ok {
			switch expectedType {
			case "string":
				if _, ok := value.(string); !ok {
					return fmt.Errorf("expected string, got %T", value)
				}
			case "number":
				switch value.(type) {
				case int, int32, int64, float32, float64:
					// Valid number types
				default:
					return fmt.Errorf("expected number, got %T", value)
				}
			case "boolean":
				if _, ok := value.(bool); !ok {
					return fmt.Errorf("expected boolean, got %T", value)
				}
			case "array":
				if _, ok := value.([]interface{}); !ok {
					return fmt.Errorf("expected array, got %T", value)
				}
			case "object":
				if _, ok := value.(map[string]interface{}); !ok {
					return fmt.Errorf("expected object, got %T", value)
				}
			}
		}
	}
	
	return nil
}

// Enhanced convenience methods for common tool patterns

// CallDatabaseTool executes a database-related tool with enhanced error handling
func (c *ProductionMCPClient) CallDatabaseTool(query string, params []interface{}) (*MCPToolResult, error) {
	if query == "" {
		return nil, fmt.Errorf("database query cannot be empty")
	}
	
	arguments := map[string]interface{}{
		"query": query,
	}
	
	if params != nil && len(params) > 0 {
		arguments["params"] = params
	}
	
	result, err := c.CallTool("database", arguments)
	if err != nil {
		return nil, fmt.Errorf("database tool execution failed: %w", err)
	}
	
	return result, nil
}

// CallFileSystemTool executes a file system operation with path validation
func (c *ProductionMCPClient) CallFileSystemTool(operation string, path string, data interface{}) (*MCPToolResult, error) {
	if operation == "" {
		return nil, fmt.Errorf("file system operation cannot be empty")
	}
	if path == "" {
		return nil, fmt.Errorf("file path cannot be empty")
	}
	
	arguments := map[string]interface{}{
		"operation": operation,
		"path":      path,
	}
	
	if data != nil {
		arguments["data"] = data
	}
	
	result, err := c.CallTool("filesystem", arguments)
	if err != nil {
		return nil, fmt.Errorf("filesystem tool execution failed: %w", err)
	}
	
	return result, nil
}

// CallAPITool executes an API call with comprehensive configuration
func (c *ProductionMCPClient) CallAPITool(endpoint string, method string, headers map[string]string, body interface{}) (*MCPToolResult, error) {
	if endpoint == "" {
		return nil, fmt.Errorf("API endpoint cannot be empty")
	}
	if method == "" {
		method = "GET"
	}
	
	arguments := map[string]interface{}{
		"endpoint": endpoint,
		"method":   method,
	}
	
	if headers != nil && len(headers) > 0 {
		arguments["headers"] = headers
	}
	
	if body != nil {
		arguments["body"] = body
	}
	
	result, err := c.CallTool("api", arguments)
	if err != nil {
		return nil, fmt.Errorf("API tool execution failed: %w", err)
	}
	
	return result, nil
}

// CallMLTool executes a machine learning operation with model validation
func (c *ProductionMCPClient) CallMLTool(operation string, modelID string, data interface{}) (*MCPToolResult, error) {
	if operation == "" {
		return nil, fmt.Errorf("ML operation cannot be empty")
	}
	if modelID == "" {
		return nil, fmt.Errorf("model ID cannot be empty")
	}
	
	arguments := map[string]interface{}{
		"operation": operation,
		"model_id":  modelID,
	}
	
	if data != nil {
		arguments["data"] = data
	}
	
	result, err := c.CallTool("ml_framework", arguments)
	if err != nil {
		return nil, fmt.Errorf("ML tool execution failed: %w", err)
	}
	
	return result, nil
}

// CallCloudTool executes cloud service operations
func (c *ProductionMCPClient) CallCloudTool(provider string, service string, action string, config map[string]interface{}) (*MCPToolResult, error) {
	if provider == "" {
		return nil, fmt.Errorf("cloud provider cannot be empty")
	}
	if service == "" {
		return nil, fmt.Errorf("cloud service cannot be empty")
	}
	if action == "" {
		return nil, fmt.Errorf("cloud action cannot be empty")
	}
	
	arguments := map[string]interface{}{
		"provider": provider,
		"service":  service,
		"action":   action,
	}
	
	if config != nil {
		arguments["config"] = config
	}
	
	toolName := fmt.Sprintf("cloud_%s", provider)
	result, err := c.CallTool(toolName, arguments)
	if err != nil {
		return nil, fmt.Errorf("cloud tool execution failed: %w", err)
	}
	
	return result, nil
}

// CallSecurityTool executes security-related operations
func (c *ProductionMCPClient) CallSecurityTool(operation string, target string, parameters map[string]interface{}) (*MCPToolResult, error) {
	if operation == "" {
		return nil, fmt.Errorf("security operation cannot be empty")
	}
	
	arguments := map[string]interface{}{
		"operation": operation,
	}
	
	if target != "" {
		arguments["target"] = target
	}
	
	if parameters != nil {
		arguments["parameters"] = parameters
	}
	
	result, err := c.CallTool("security", arguments)
	if err != nil {
		return nil, fmt.Errorf("security tool execution failed: %w", err)
	}
	
	return result, nil
}

// CallEnterpriseIntegrationTool executes enterprise system integrations
func (c *ProductionMCPClient) CallEnterpriseIntegrationTool(system string, operation string, credentials map[string]string, payload interface{}) (*MCPToolResult, error) {
	if system == "" {
		return nil, fmt.Errorf("enterprise system cannot be empty")
	}
	if operation == "" {
		return nil, fmt.Errorf("operation cannot be empty")
	}
	
	arguments := map[string]interface{}{
		"system":    system,
		"operation": operation,
	}
	
	if credentials != nil {
		arguments["credentials"] = credentials
	}
	
	if payload != nil {
		arguments["payload"] = payload
	}
	
	toolName := fmt.Sprintf("enterprise_%s", system)
	result, err := c.CallTool(toolName, arguments)
	if err != nil {
		return nil, fmt.Errorf("enterprise integration tool execution failed: %w", err)
	}
	
	return result, nil
}

// Batch tool execution for enterprise efficiency

// BatchToolCalls executes multiple tools in parallel
func (c *ProductionMCPClient) BatchToolCalls(calls []ToolCall) ([]ToolCallResult, error) {
	if len(calls) == 0 {
		return []ToolCallResult{}, nil
	}
	
	results := make([]ToolCallResult, len(calls))
	var wg sync.WaitGroup
	
	for i, call := range calls {
		wg.Add(1)
		go func(index int, toolCall ToolCall) {
			defer wg.Done()
			
			result, err := c.CallTool(toolCall.Name, toolCall.Arguments)
			results[index] = ToolCallResult{
				Index:  index,
				Name:   toolCall.Name,
				Result: result,
				Error:  err,
			}
		}(i, call)
	}
	
	wg.Wait()
	return results, nil
}

// Sequential tool execution with dependency handling
func (c *ProductionMCPClient) SequentialToolCalls(calls []ToolCall, shareContext bool) ([]ToolCallResult, error) {
	results := make([]ToolCallResult, len(calls))
	context := make(map[string]interface{})
	
	for i, call := range calls {
		arguments := call.Arguments
		
		// Merge context if sharing is enabled
		if shareContext && i > 0 {
			if argMap, ok := arguments.(map[string]interface{}); ok {
				// Add previous results to context
				for j := 0; j < i; j++ {
					if results[j].Result != nil && !results[j].Result.IsError {
						contextKey := fmt.Sprintf("prev_result_%d", j)
						context[contextKey] = results[j].Result
					}
				}
				
				// Merge context into arguments
				for k, v := range context {
					if _, exists := argMap[k]; !exists {
						argMap[k] = v
					}
				}
				arguments = argMap
			}
		}
		
		result, err := c.CallTool(call.Name, arguments)
		results[i] = ToolCallResult{
			Index:  i,
			Name:   call.Name,
			Result: result,
			Error:  err,
		}
		
		// Stop on error if specified
		if err != nil && call.StopOnError {
			break
		}
	}
	
	return results, nil
}

// Tool execution types
type ToolCall struct {
	Name        string      `json:"name"`
	Arguments   interface{} `json:"arguments"`
	StopOnError bool        `json:"stop_on_error,omitempty"`
}

type ToolCallResult struct {
	Index  int            `json:"index"`
	Name   string         `json:"name"`
	Result *MCPToolResult `json:"result,omitempty"`
	Error  error          `json:"error,omitempty"`
}