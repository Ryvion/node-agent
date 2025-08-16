// File: node-agent/internal/mcp/resources_production.go
package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"time"
)

// ReadResource reads content from a specific resource URI
func (c *ProductionMCPClient) ReadResource(uri string) (*MCPResourceContent, error) {
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
	
	// Verify resource exists or matches template
	if !c.isValidResourceURI(uri) {
		return nil, fmt.Errorf("invalid or unknown resource URI: %s", uri)
	}
	
	// Create resource read message
	readMsg := &MCPMessage{
		JSONRPCVersion: "2.0",
		ID:             c.nextRequestID(),
		Method:         "resources/read",
		Params: map[string]interface{}{
			"uri": uri,
		},
	}
	
	c.logger.Printf("Reading resource: %s", uri)
	
	// Send request and wait for response
	response, err := c.sendRequestWithResponse(readMsg)
	if err != nil {
		c.metrics.FailedRequests++
		return nil, fmt.Errorf("resource read failed: %w", err)
	}
	
	if response.Error != nil {
		c.metrics.FailedRequests++
		return nil, fmt.Errorf("resource read error: %s", response.Error.Message)
	}
	
	// Parse resource content
	if result, ok := response.Result.(map[string]interface{}); ok {
		resultBytes, _ := json.Marshal(result)
		var resourceContent MCPResourceContent
		if err := json.Unmarshal(resultBytes, &resourceContent); err != nil {
			c.metrics.FailedRequests++
			return nil, fmt.Errorf("failed to parse resource content: %w", err)
		}
		
		c.metrics.SuccessfulRequests++
		c.metrics.ResourcesAccessed++
		c.logger.Printf("Successfully read resource '%s' (%d content items)", uri, len(resourceContent.Contents))
		return &resourceContent, nil
	}
	
	c.metrics.FailedRequests++
	return nil, fmt.Errorf("invalid resource content format")
}

// ReadResourceAsync reads a resource asynchronously
func (c *ProductionMCPClient) ReadResourceAsync(uri string) (<-chan *MCPResourceContent, <-chan error) {
	resultChan := make(chan *MCPResourceContent, 1)
	errorChan := make(chan error, 1)
	
	go func() {
		defer close(resultChan)
		defer close(errorChan)
		
		result, err := c.ReadResource(uri)
		if err != nil {
			errorChan <- err
			return
		}
		
		resultChan <- result
	}()
	
	return resultChan, errorChan
}

// ReadResourceWithTimeout reads a resource with custom timeout
func (c *ProductionMCPClient) ReadResourceWithTimeout(uri string, timeout time.Duration) (*MCPResourceContent, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	
	resultChan, errorChan := c.ReadResourceAsync(uri)
	
	select {
	case result := <-resultChan:
		return result, nil
	case err := <-errorChan:
		return nil, err
	case <-ctx.Done():
		return nil, fmt.Errorf("resource read timeout after %v", timeout)
	}
}

// ReadResourceWithContext reads a resource with the given context
func (c *ProductionMCPClient) ReadResourceWithContext(ctx context.Context, uri string) (*MCPResourceContent, error) {
	type result struct {
		content *MCPResourceContent
		err     error
	}
	
	resultChan := make(chan result, 1)
	
	go func() {
		content, err := c.ReadResource(uri)
		resultChan <- result{content: content, err: err}
	}()
	
	select {
	case res := <-resultChan:
		return res.content, res.err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// SubscribeToResource subscribes to resource changes (if supported)
func (c *ProductionMCPClient) SubscribeToResource(uri string) error {
	if !c.IsConnected() {
		return fmt.Errorf("client not connected")
	}
	
	// Check if server supports resource subscriptions
	if c.serverInfo == nil || c.serverInfo.Capabilities.Resources == nil || !c.serverInfo.Capabilities.Resources.Subscribe {
		return fmt.Errorf("server does not support resource subscriptions")
	}
	
	subscribeMsg := &MCPMessage{
		JSONRPCVersion: "2.0",
		ID:             c.nextRequestID(),
		Method:         "resources/subscribe",
		Params: map[string]interface{}{
			"uri": uri,
		},
	}
	
	response, err := c.sendRequestWithResponse(subscribeMsg)
	if err != nil {
		return fmt.Errorf("resource subscription failed: %w", err)
	}
	
	if response.Error != nil {
		return fmt.Errorf("resource subscription error: %s", response.Error.Message)
	}
	
	c.logger.Printf("Subscribed to resource: %s", uri)
	return nil
}

// UnsubscribeFromResource unsubscribes from resource changes
func (c *ProductionMCPClient) UnsubscribeFromResource(uri string) error {
	if !c.IsConnected() {
		return fmt.Errorf("client not connected")
	}
	
	unsubscribeMsg := &MCPMessage{
		JSONRPCVersion: "2.0",
		ID:             c.nextRequestID(),
		Method:         "resources/unsubscribe",
		Params: map[string]interface{}{
			"uri": uri,
		},
	}
	
	response, err := c.sendRequestWithResponse(unsubscribeMsg)
	if err != nil {
		return fmt.Errorf("resource unsubscription failed: %w", err)
	}
	
	if response.Error != nil {
		return fmt.Errorf("resource unsubscription error: %s", response.Error.Message)
	}
	
	c.logger.Printf("Unsubscribed from resource: %s", uri)
	return nil
}

// Helper method to validate resource URI
func (c *ProductionMCPClient) isValidResourceURI(uri string) bool {
	c.dataMutex.RLock()
	defer c.dataMutex.RUnlock()
	
	// Check direct resource match
	if _, exists := c.resources[uri]; exists {
		return true
	}
	
	// Check against resource templates
	for templateURI := range c.resources {
		if c.matchesResourceTemplate(templateURI, uri) {
			return true
		}
	}
	
	return false
}

// Check if URI matches a resource template pattern
func (c *ProductionMCPClient) matchesResourceTemplate(template, uri string) bool {
	// Enhanced template matching with better pattern support
	if !contains(template, "{") {
		return template == uri
	}
	
	return c.advancedTemplateMatch(template, uri)
}

// advancedTemplateMatch implements sophisticated template matching
func (c *ProductionMCPClient) advancedTemplateMatch(template, uri string) bool {
	// Split by forward slashes for path-like URIs
	templateParts := splitURI(template)
	uriParts := splitURI(uri)
	
	if len(templateParts) != len(uriParts) {
		return false
	}
	
	for i, templatePart := range templateParts {
		if isTemplateVariable(templatePart) {
			// This is a variable part, so it matches any value
			continue
		}
		
		if templatePart != uriParts[i] {
			return false
		}
	}
	
	return true
}

// splitURI splits a URI into parts for template matching
func splitURI(uri string) []string {
	// Handle different URI schemes
	if contains(uri, "://") {
		// Split by :// first, then handle the rest
		parts := []string{}
		schemeParts := []string{uri[:len("scheme")], uri[len("scheme://"):]}
		if len(schemeParts) == 2 {
			parts = append(parts, schemeParts[0]+"://")
			remainingParts := []string{}
			remaining := schemeParts[1]
			for i := 0; i < len(remaining); i++ {
				if remaining[i] == '/' {
					if len(remainingParts) > 0 {
						parts = append(parts, remainingParts...)
						remainingParts = []string{}
					}
					parts = append(parts, "/")
				} else {
					// Add character to current part
					if len(remainingParts) == 0 {
						remainingParts = append(remainingParts, string(remaining[i]))
					} else {
						remainingParts[len(remainingParts)-1] += string(remaining[i])
					}
				}
			}
			if len(remainingParts) > 0 {
				parts = append(parts, remainingParts...)
			}
		}
		return parts
	}
	
	// Simple split by /
	parts := []string{}
	current := ""
	for i := 0; i < len(uri); i++ {
		if uri[i] == '/' {
			if current != "" {
				parts = append(parts, current)
				current = ""
			}
		} else {
			current += string(uri[i])
		}
	}
	if current != "" {
		parts = append(parts, current)
	}
	
	return parts
}

// isTemplateVariable checks if a part is a template variable
func isTemplateVariable(part string) bool {
	return len(part) >= 3 && part[0] == '{' && part[len(part)-1] == '}'
}

// Convenience methods for common resource patterns

// ReadDatabaseResource reads data from a database resource
func (c *ProductionMCPClient) ReadDatabaseResource(database, table string, query map[string]interface{}) (*MCPResourceContent, error) {
	if database == "" || table == "" {
		return nil, fmt.Errorf("database and table names cannot be empty")
	}
	
	uri := fmt.Sprintf("database://%s/%s", database, table)
	if query != nil && len(query) > 0 {
		// Add query parameters - simplified implementation
		uri += "?query=custom"
	}
	
	return c.ReadResource(uri)
}

// ReadFileResource reads content from a file resource
func (c *ProductionMCPClient) ReadFileResource(filepath string) (*MCPResourceContent, error) {
	if filepath == "" {
		return nil, fmt.Errorf("file path cannot be empty")
	}
	
	uri := fmt.Sprintf("file://%s", filepath)
	return c.ReadResource(uri)
}

// ReadAPIResource reads data from an API endpoint resource
func (c *ProductionMCPClient) ReadAPIResource(endpoint string, params map[string]string) (*MCPResourceContent, error) {
	if endpoint == "" {
		return nil, fmt.Errorf("API endpoint cannot be empty")
	}
	
	uri := endpoint
	if params != nil && len(params) > 0 {
		// Add query parameters - simplified implementation
		uri += "?params=custom"
	}
	
	return c.ReadResource(uri)
}

// ReadCloudStorageResource reads from cloud storage (S3, GCS, etc.)
func (c *ProductionMCPClient) ReadCloudStorageResource(provider, bucket, key string) (*MCPResourceContent, error) {
	if provider == "" || bucket == "" || key == "" {
		return nil, fmt.Errorf("provider, bucket, and key cannot be empty")
	}
	
	uri := fmt.Sprintf("%s://%s/%s", provider, bucket, key)
	return c.ReadResource(uri)
}

// Batch resource operations

// BatchReadResources reads multiple resources in parallel
func (c *ProductionMCPClient) BatchReadResources(uris []string) (map[string]*MCPResourceContent, map[string]error) {
	if len(uris) == 0 {
		return make(map[string]*MCPResourceContent), make(map[string]error)
	}
	
	results := make(map[string]*MCPResourceContent)
	errors := make(map[string]error)
	
	type result struct {
		uri     string
		content *MCPResourceContent
		err     error
	}
	
	resultChan := make(chan result, len(uris))
	
	// Start all reads in parallel
	for _, uri := range uris {
		go func(u string) {
			content, err := c.ReadResource(u)
			resultChan <- result{uri: u, content: content, err: err}
		}(uri)
	}
	
	// Collect results
	for i := 0; i < len(uris); i++ {
		res := <-resultChan
		if res.err != nil {
			errors[res.uri] = res.err
		} else {
			results[res.uri] = res.content
		}
	}
	
	return results, errors
}