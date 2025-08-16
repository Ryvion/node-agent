// File: node-agent/internal/mcp/types.go
package mcp

import (
	"time"
)

// MCP Protocol Version
const MCPVersion = "2024-11-05"

// Core MCP message types
type MCPMessage struct {
	JSONRPCVersion string      `json:"jsonrpc"`
	ID             interface{} `json:"id,omitempty"`
	Method         string      `json:"method,omitempty"`
	Params         interface{} `json:"params,omitempty"`
	Result         interface{} `json:"result,omitempty"`
	Error          *MCPError   `json:"error,omitempty"`
}

type MCPError struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

// MCP Server Information
type MCPServerInfo struct {
	Name         string            `json:"name"`
	Version      string            `json:"version"`
	Protocol     string            `json:"protocol"`
	Capabilities MCPCapabilities   `json:"capabilities"`
	Instructions string            `json:"instructions,omitempty"`
	Metadata     map[string]string `json:"metadata,omitempty"`
}

type MCPCapabilities struct {
	Tools     *MCPToolCapabilities     `json:"tools,omitempty"`
	Resources *MCPResourceCapabilities `json:"resources,omitempty"`
	Prompts   *MCPPromptCapabilities   `json:"prompts,omitempty"`
	Logging   *MCPLoggingCapabilities  `json:"logging,omitempty"`
}

type MCPToolCapabilities struct {
	ListChanged bool `json:"listChanged,omitempty"`
}

type MCPResourceCapabilities struct {
	Subscribe   bool `json:"subscribe,omitempty"`
	ListChanged bool `json:"listChanged,omitempty"`
}

type MCPPromptCapabilities struct {
	ListChanged bool `json:"listChanged,omitempty"`
}

type MCPLoggingCapabilities struct {
	Level string `json:"level,omitempty"`
}

// Tool definitions
type MCPTool struct {
	Name        string            `json:"name"`
	Description string            `json:"description"`
	InputSchema MCPToolSchema     `json:"inputSchema"`
	Examples    []MCPToolExample  `json:"examples,omitempty"`
}

type MCPToolSchema struct {
	Type       string                 `json:"type"`
	Properties map[string]interface{} `json:"properties,omitempty"`
	Required   []string               `json:"required,omitempty"`
}

type MCPToolExample struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Arguments   interface{} `json:"arguments"`
}

type MCPToolCall struct {
	Name      string      `json:"name"`
	Arguments interface{} `json:"arguments"`
}

type MCPToolResult struct {
	Content []MCPContent `json:"content"`
	IsError bool         `json:"isError,omitempty"`
}

// Resource definitions
type MCPResource struct {
	URI         string `json:"uri"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	MimeType    string `json:"mimeType,omitempty"`
}

type MCPResourceTemplate struct {
	URITemplate string `json:"uriTemplate"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	MimeType    string `json:"mimeType,omitempty"`
}

// Content types
type MCPContent struct {
	Type        string      `json:"type"`
	Text        string      `json:"text,omitempty"`
	Data        string      `json:"data,omitempty"`
	MimeType    string      `json:"mimeType,omitempty"`
	Annotations interface{} `json:"annotations,omitempty"`
}

// Prompt definitions
type MCPPrompt struct {
	Name        string               `json:"name"`
	Description string               `json:"description,omitempty"`
	Arguments   []MCPPromptArgument  `json:"arguments,omitempty"`
}

type MCPPromptArgument struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Required    bool   `json:"required,omitempty"`
}

type MCPPromptMessage struct {
	Role    string       `json:"role"`
	Content []MCPContent `json:"content"`
}

// Logging
type MCPLogEntry struct {
	Level  string      `json:"level"`
	Data   interface{} `json:"data"`
	Logger string      `json:"logger,omitempty"`
}

// Progress reporting
type MCPProgress struct {
	Progress int    `json:"progress"`
	Total    int    `json:"total,omitempty"`
	Token    string `json:"token"`
}

// Connection states
type MCPConnectionState int

const (
	MCPStateDisconnected MCPConnectionState = iota
	MCPStateConnecting
	MCPStateConnected
	MCPStateError
)

// Authentication
type MCPAuthRequest struct {
	Type string                 `json:"type"`
	Data map[string]interface{} `json:"data"`
}

type MCPAuthResponse struct {
	Authorized bool   `json:"authorized"`
	Token      string `json:"token,omitempty"`
	ExpiresAt  int64  `json:"expiresAt,omitempty"`
}

// Additional types for enhanced functionality
// (MCPResourceContent and MCPPromptResult are defined in their respective files)

// Client configuration
type MCPClientConfig struct {
	ServerURL    string
	AuthToken    string
	Timeout      time.Duration
	RetryDelay   time.Duration
	MaxRetries   int
	Logger       interface{} // Using interface{} to avoid import cycles
}

// MCPResourceContent represents the content returned from a resource
type MCPResourceContent struct {
	Contents []MCPContent           `json:"contents"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// GetTextContent extracts text content from resource contents
func (rc *MCPResourceContent) GetTextContent() []string {
	var textContents []string
	for _, content := range rc.Contents {
		if content.Type == "text" && content.Text != "" {
			textContents = append(textContents, content.Text)
		}
	}
	return textContents
}

// GetDataContent extracts binary data content from resource contents
func (rc *MCPResourceContent) GetDataContent() [][]byte {
	var dataContents [][]byte
	for _, content := range rc.Contents {
		if content.Type == "blob" && content.Data != "" {
			dataContents = append(dataContents, []byte(content.Data))
		}
	}
	return dataContents
}

// GetMetadata returns resource metadata
func (rc *MCPResourceContent) GetMetadata(key string) interface{} {
	if rc.Metadata == nil {
		return nil
	}
	return rc.Metadata[key]
}

// MCPPromptResult represents the result from a prompt request
type MCPPromptResult struct {
	Description string             `json:"description,omitempty"`
	Messages    []MCPPromptMessage `json:"messages"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// GetSystemMessage extracts system messages from prompt result
func (pr *MCPPromptResult) GetSystemMessage() *MCPPromptMessage {
	for _, message := range pr.Messages {
		if message.Role == "system" {
			return &message
		}
	}
	return nil
}

// GetUserMessages extracts user messages from prompt result
func (pr *MCPPromptResult) GetUserMessages() []MCPPromptMessage {
	var userMessages []MCPPromptMessage
	for _, message := range pr.Messages {
		if message.Role == "user" {
			userMessages = append(userMessages, message)
		}
	}
	return userMessages
}

// GetAssistantMessages extracts assistant messages from prompt result
func (pr *MCPPromptResult) GetAssistantMessages() []MCPPromptMessage {
	var assistantMessages []MCPPromptMessage
	for _, message := range pr.Messages {
		if message.Role == "assistant" {
			assistantMessages = append(assistantMessages, message)
		}
	}
	return assistantMessages
}

// GetTextContent extracts all text content from messages
func (pr *MCPPromptResult) GetTextContent() []string {
	var textContents []string
	for _, message := range pr.Messages {
		for _, content := range message.Content {
			if content.Type == "text" && content.Text != "" {
				textContents = append(textContents, content.Text)
			}
		}
	}
	return textContents
}

// GetCombinedText combines all text content into a single string
func (pr *MCPPromptResult) GetCombinedText() string {
	textContents := pr.GetTextContent()
	combinedText := ""
	for _, text := range textContents {
		if combinedText != "" {
			combinedText += "\n"
		}
		combinedText += text
	}
	return combinedText
}

// GetMetadata returns prompt metadata
func (pr *MCPPromptResult) GetMetadata(key string) interface{} {
	if pr.Metadata == nil {
		return nil
	}
	return pr.Metadata[key]
}

// Helper function for string contains
func contains(s, substr string) bool {
	return len(s) >= len(substr) && 
		   (s == substr || 
		    s[:len(substr)] == substr ||
		    s[len(s)-len(substr):] == substr ||
		    containsSubstring(s, substr))
}

func containsSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}