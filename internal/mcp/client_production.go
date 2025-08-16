// File: node-agent/internal/mcp/client_production.go
package mcp

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
)

// ProductionMCPClient is a production-ready MCP client implementation
type ProductionMCPClient struct {
	// Connection management
	serverURL       string
	conn            *websocket.Conn
	connMutex       sync.RWMutex
	state           int32 // atomic access to MCPConnectionState
	requestID       int64 // atomic counter for request IDs
	
	// Authentication & Security
	authToken       string
	tlsConfig       *tls.Config
	
	// Server capabilities and data
	serverInfo      *MCPServerInfo
	tools           map[string]*MCPTool
	resources       map[string]*MCPResource
	prompts         map[string]*MCPPrompt
	dataMutex       sync.RWMutex
	
	// Request/Response handling
	pendingRequests map[string]chan *MCPMessage
	requestMutex    sync.RWMutex
	
	// Event handlers
	eventHandlers   MCPEventHandlers
	
	// Configuration
	config          ProductionMCPConfig
	
	// Logging
	logger          *log.Logger
	
	// Context management
	ctx             context.Context
	cancel          context.CancelFunc
	
	// Message channels
	incomingMessages chan *MCPMessage
	outgoingMessages chan *MCPMessage
	errorChannel     chan error
	
	// Connection health
	lastPong        time.Time
	pingInterval    time.Duration
	pongTimeout     time.Duration
	
	// Reconnection
	reconnectAttempts int
	maxReconnects     int
	reconnectDelay    time.Duration
	
	// Metrics and monitoring
	metrics         *MCPClientMetrics
	metricsCallback func(*MCPClientMetrics)
}

type ProductionMCPConfig struct {
	ServerURL           string
	AuthToken           string
	TLSConfig           *tls.Config
	RequestTimeout      time.Duration
	PingInterval        time.Duration
	PongTimeout         time.Duration
	MaxReconnects       int
	ReconnectDelay      time.Duration
	MaxMessageSize      int64
	ReadBufferSize      int
	WriteBufferSize     int
	EnableCompression   bool
	Logger              *log.Logger
}

type MCPEventHandlers struct {
	OnConnected           func()
	OnDisconnected        func(error)
	OnToolListChanged     func()
	OnResourceListChanged func()
	OnPromptListChanged   func()
	OnLogEntry           func(*MCPLogEntry)
	OnResourceUpdated    func(string, string, *MCPResourceContent)
	OnError              func(error)
}

type MCPClientMetrics struct {
	ConnectedAt         time.Time
	LastActivity        time.Time
	TotalRequests       int64
	SuccessfulRequests  int64
	FailedRequests      int64
	AverageResponseTime time.Duration
	ReconnectCount      int
	MessagesReceived    int64
	MessagesSent        int64
	ToolCallsExecuted   int64
	ResourcesAccessed   int64
	PromptsRetrieved    int64
}

// NewProductionMCPClient creates a new production-ready MCP client
func NewProductionMCPClient(config ProductionMCPConfig) *ProductionMCPClient {
	// Set defaults
	if config.RequestTimeout == 0 {
		config.RequestTimeout = 30 * time.Second
	}
	if config.PingInterval == 0 {
		config.PingInterval = 30 * time.Second
	}
	if config.PongTimeout == 0 {
		config.PongTimeout = 10 * time.Second
	}
	if config.MaxReconnects == 0 {
		config.MaxReconnects = 5
	}
	if config.ReconnectDelay == 0 {
		config.ReconnectDelay = 5 * time.Second
	}
	if config.MaxMessageSize == 0 {
		config.MaxMessageSize = 1024 * 1024 // 1MB
	}
	if config.ReadBufferSize == 0 {
		config.ReadBufferSize = 4096
	}
	if config.WriteBufferSize == 0 {
		config.WriteBufferSize = 4096
	}
	if config.Logger == nil {
		config.Logger = log.New(log.Writer(), "[MCP] ", log.LstdFlags|log.Lshortfile)
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	return &ProductionMCPClient{
		serverURL:         config.ServerURL,
		authToken:         config.AuthToken,
		tlsConfig:         config.TLSConfig,
		state:            int32(MCPStateDisconnected),
		tools:            make(map[string]*MCPTool),
		resources:        make(map[string]*MCPResource),
		prompts:          make(map[string]*MCPPrompt),
		pendingRequests:  make(map[string]chan *MCPMessage),
		config:           config,
		logger:           config.Logger,
		ctx:              ctx,
		cancel:           cancel,
		incomingMessages: make(chan *MCPMessage, 100),
		outgoingMessages: make(chan *MCPMessage, 100),
		errorChannel:     make(chan error, 10),
		pingInterval:     config.PingInterval,
		pongTimeout:      config.PongTimeout,
		maxReconnects:    config.MaxReconnects,
		reconnectDelay:   config.ReconnectDelay,
		metrics:          &MCPClientMetrics{},
	}
}

// Connect establishes connection to the MCP server
func (c *ProductionMCPClient) Connect() error {
	if !atomic.CompareAndSwapInt32(&c.state, int32(MCPStateDisconnected), int32(MCPStateConnecting)) {
		return fmt.Errorf("client already connecting or connected")
	}
	
	c.logger.Printf("Connecting to MCP server: %s", c.serverURL)
	
	// Parse and validate URL
	u, err := url.Parse(c.serverURL)
	if err != nil {
		atomic.StoreInt32(&c.state, int32(MCPStateError))
		return fmt.Errorf("invalid server URL: %w", err)
	}
	
	// Configure WebSocket dialer
	dialer := &websocket.Dialer{
		HandshakeTimeout: c.config.RequestTimeout,
		ReadBufferSize:   c.config.ReadBufferSize,
		WriteBufferSize:  c.config.WriteBufferSize,
		TLSClientConfig:  c.tlsConfig,
		EnableCompression: c.config.EnableCompression,
	}
	
	// Set up headers
	headers := http.Header{}
	if c.authToken != "" {
		headers.Set("Authorization", "Bearer "+c.authToken)
	}
	headers.Set("Sec-WebSocket-Protocol", "mcp")
	
	// Establish WebSocket connection
	conn, resp, err := dialer.Dial(u.String(), headers)
	if err != nil {
		atomic.StoreInt32(&c.state, int32(MCPStateError))
		if resp != nil {
			return fmt.Errorf("WebSocket connection failed (%d): %w", resp.StatusCode, err)
		}
		return fmt.Errorf("WebSocket connection failed: %w", err)
	}
	
	// Configure connection
	conn.SetReadLimit(c.config.MaxMessageSize)
	conn.SetReadDeadline(time.Now().Add(c.pongTimeout))
	conn.SetPongHandler(func(string) error {
		c.lastPong = time.Now()
		conn.SetReadDeadline(time.Now().Add(c.pongTimeout))
		return nil
	})
	
	c.connMutex.Lock()
	c.conn = conn
	c.connMutex.Unlock()
	
	atomic.StoreInt32(&c.state, int32(MCPStateConnected))
	c.metrics.ConnectedAt = time.Now()
	c.metrics.LastActivity = time.Now()
	c.lastPong = time.Now()
	
	// Start message processing goroutines
	go c.messageReader()
	go c.messageWriter()
	go c.messageProcessor()
	go c.pingHandler()
	
	// Initialize MCP protocol
	if err := c.initializeMCP(); err != nil {
		c.Close()
		return fmt.Errorf("MCP initialization failed: %w", err)
	}
	
	c.logger.Printf("Successfully connected to MCP server")
	
	// Trigger connected event
	if c.eventHandlers.OnConnected != nil {
		go c.eventHandlers.OnConnected()
	}
	
	return nil
}

// initializeMCP performs the MCP protocol initialization handshake
func (c *ProductionMCPClient) initializeMCP() error {
	c.logger.Printf("Initializing MCP protocol...")
	
	// Send initialize request
	initRequest := &MCPMessage{
		JSONRPCVersion: "2.0",
		ID:             c.nextRequestID(),
		Method:         "initialize",
		Params: map[string]interface{}{
			"protocolVersion": MCPVersion,
			"capabilities": map[string]interface{}{
				"tools": map[string]interface{}{
					"listChanged": true,
				},
				"resources": map[string]interface{}{
					"subscribe":   true,
					"listChanged": true,
				},
				"prompts": map[string]interface{}{
					"listChanged": true,
				},
			},
			"clientInfo": map[string]interface{}{
				"name":    "ryvion-production-agent",
				"version": "1.0.0",
			},
		},
	}
	
	response, err := c.sendRequestWithResponse(initRequest)
	if err != nil {
		return fmt.Errorf("initialize request failed: %w", err)
	}
	
	if response.Error != nil {
		return fmt.Errorf("initialize failed: %s", response.Error.Message)
	}
	
	// Parse server info
	if result, ok := response.Result.(map[string]interface{}); ok {
		serverInfoBytes, _ := json.Marshal(result)
		if err := json.Unmarshal(serverInfoBytes, &c.serverInfo); err != nil {
			c.logger.Printf("Warning: failed to parse server info: %v", err)
		} else {
			c.logger.Printf("Connected to MCP server: %s v%s", 
				c.serverInfo.Name, c.serverInfo.Version)
		}
	}
	
	// Send initialized notification
	initializedNotification := &MCPMessage{
		JSONRPCVersion: "2.0",
		Method:         "notifications/initialized",
		Params:         map[string]interface{}{},
	}
	
	if err := c.sendMessage(initializedNotification); err != nil {
		return fmt.Errorf("initialized notification failed: %w", err)
	}
	
	// Load initial capabilities
	go func() {
		if err := c.loadTools(); err != nil {
			c.logger.Printf("Warning: failed to load tools: %v", err)
		}
		if err := c.loadResources(); err != nil {
			c.logger.Printf("Warning: failed to load resources: %v", err)
		}
		if err := c.loadPrompts(); err != nil {
			c.logger.Printf("Warning: failed to load prompts: %v", err)
		}
	}()
	
	c.logger.Printf("MCP protocol initialization completed")
	return nil
}

// loadTools loads available tools from the server
func (c *ProductionMCPClient) loadTools() error {
	listRequest := &MCPMessage{
		JSONRPCVersion: "2.0",
		ID:             c.nextRequestID(),
		Method:         "tools/list",
		Params:         map[string]interface{}{},
	}
	
	response, err := c.sendRequestWithResponse(listRequest)
	if err != nil {
		return err
	}
	
	if response.Error != nil {
		return fmt.Errorf("tools/list failed: %s", response.Error.Message)
	}
	
	// Parse tools
	if result, ok := response.Result.(map[string]interface{}); ok {
		if toolsData, ok := result["tools"].([]interface{}); ok {
			c.dataMutex.Lock()
			c.tools = make(map[string]*MCPTool)
			
			for _, toolData := range toolsData {
				toolBytes, _ := json.Marshal(toolData)
				var tool MCPTool
				if err := json.Unmarshal(toolBytes, &tool); err == nil {
					c.tools[tool.Name] = &tool
				}
			}
			c.dataMutex.Unlock()
			
			c.logger.Printf("Loaded %d tools from MCP server", len(c.tools))
		}
	}
	
	return nil
}

// loadResources loads available resources from the server
func (c *ProductionMCPClient) loadResources() error {
	listRequest := &MCPMessage{
		JSONRPCVersion: "2.0",
		ID:             c.nextRequestID(),
		Method:         "resources/list",
		Params:         map[string]interface{}{},
	}
	
	response, err := c.sendRequestWithResponse(listRequest)
	if err != nil {
		return err
	}
	
	if response.Error != nil {
		return fmt.Errorf("resources/list failed: %s", response.Error.Message)
	}
	
	// Parse resources
	if result, ok := response.Result.(map[string]interface{}); ok {
		if resourcesData, ok := result["resources"].([]interface{}); ok {
			c.dataMutex.Lock()
			c.resources = make(map[string]*MCPResource)
			
			for _, resourceData := range resourcesData {
				resourceBytes, _ := json.Marshal(resourceData)
				var resource MCPResource
				if err := json.Unmarshal(resourceBytes, &resource); err == nil {
					c.resources[resource.URI] = &resource
				}
			}
			c.dataMutex.Unlock()
			
			c.logger.Printf("Loaded %d resources from MCP server", len(c.resources))
		}
	}
	
	return nil
}

// loadPrompts loads available prompts from the server
func (c *ProductionMCPClient) loadPrompts() error {
	listRequest := &MCPMessage{
		JSONRPCVersion: "2.0",
		ID:             c.nextRequestID(),
		Method:         "prompts/list",
		Params:         map[string]interface{}{},
	}
	
	response, err := c.sendRequestWithResponse(listRequest)
	if err != nil {
		return err
	}
	
	if response.Error != nil {
		return fmt.Errorf("prompts/list failed: %s", response.Error.Message)
	}
	
	// Parse prompts
	if result, ok := response.Result.(map[string]interface{}); ok {
		if promptsData, ok := result["prompts"].([]interface{}); ok {
			c.dataMutex.Lock()
			c.prompts = make(map[string]*MCPPrompt)
			
			for _, promptData := range promptsData {
				promptBytes, _ := json.Marshal(promptData)
				var prompt MCPPrompt
				if err := json.Unmarshal(promptBytes, &prompt); err == nil {
					c.prompts[prompt.Name] = &prompt
				}
			}
			c.dataMutex.Unlock()
			
			c.logger.Printf("Loaded %d prompts from MCP server", len(c.prompts))
		}
	}
	
	return nil
}

// Message handling goroutines

// messageReader reads messages from the WebSocket connection
func (c *ProductionMCPClient) messageReader() {
	defer func() {
		if r := recover(); r != nil {
			c.logger.Printf("Message reader panic: %v", r)
		}
		c.handleDisconnection(fmt.Errorf("message reader stopped"))
	}()
	
	for {
		select {
		case <-c.ctx.Done():
			return
		default:
			var msg MCPMessage
			
			c.connMutex.RLock()
			conn := c.conn
			c.connMutex.RUnlock()
			
			if conn == nil {
				return
			}
			
			if err := conn.ReadJSON(&msg); err != nil {
				if !websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					c.logger.Printf("WebSocket read error: %v", err)
				}
				c.errorChannel <- fmt.Errorf("read message error: %w", err)
				return
			}
			
			c.metrics.MessagesReceived++
			c.metrics.LastActivity = time.Now()
			
			select {
			case c.incomingMessages <- &msg:
			case <-c.ctx.Done():
				return
			}
		}
	}
}

// messageWriter writes messages to the WebSocket connection
func (c *ProductionMCPClient) messageWriter() {
	defer func() {
		if r := recover(); r != nil {
			c.logger.Printf("Message writer panic: %v", r)
		}
	}()
	
	for {
		select {
		case <-c.ctx.Done():
			return
		case msg := <-c.outgoingMessages:
			c.connMutex.RLock()
			conn := c.conn
			c.connMutex.RUnlock()
			
			if conn == nil {
				continue
			}
			
			if err := conn.WriteJSON(msg); err != nil {
				c.logger.Printf("WebSocket write error: %v", err)
				c.errorChannel <- fmt.Errorf("write message error: %w", err)
				return
			}
			
			c.metrics.MessagesSent++
			c.metrics.LastActivity = time.Now()
		}
	}
}

// messageProcessor processes incoming messages
func (c *ProductionMCPClient) messageProcessor() {
	defer func() {
		if r := recover(); r != nil {
			c.logger.Printf("Message processor panic: %v", r)
		}
	}()
	
	for {
		select {
		case <-c.ctx.Done():
			return
		case err := <-c.errorChannel:
			c.logger.Printf("MCP client error: %v", err)
			c.handleDisconnection(err)
			return
		case msg := <-c.incomingMessages:
			c.handleMessage(msg)
		}
	}
}

// pingHandler sends periodic ping messages to keep connection alive
func (c *ProductionMCPClient) pingHandler() {
	ticker := time.NewTicker(c.pingInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-c.ctx.Done():
			return
		case <-ticker.C:
			if atomic.LoadInt32(&c.state) != int32(MCPStateConnected) {
				continue
			}
			
			c.connMutex.RLock()
			conn := c.conn
			c.connMutex.RUnlock()
			
			if conn == nil {
				continue
			}
			
			if err := conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				c.logger.Printf("Ping failed: %v", err)
				c.errorChannel <- fmt.Errorf("ping failed: %w", err)
				return
			}
			
			// Check if we received a pong recently
			if time.Since(c.lastPong) > c.pongTimeout {
				c.logger.Printf("Pong timeout exceeded")
				c.errorChannel <- fmt.Errorf("pong timeout")
				return
			}
		}
	}
}

// nextRequestID generates a unique request ID
func (c *ProductionMCPClient) nextRequestID() int64 {
	return atomic.AddInt64(&c.requestID, 1)
}

// sendMessage sends a message to the server
func (c *ProductionMCPClient) sendMessage(msg *MCPMessage) error {
	if atomic.LoadInt32(&c.state) != int32(MCPStateConnected) {
		return fmt.Errorf("client not connected")
	}
	
	select {
	case c.outgoingMessages <- msg:
		return nil
	case <-time.After(c.config.RequestTimeout):
		return fmt.Errorf("message send timeout")
	case <-c.ctx.Done():
		return fmt.Errorf("client disconnected")
	}
}

// sendRequestWithResponse sends a request and waits for the response
func (c *ProductionMCPClient) sendRequestWithResponse(msg *MCPMessage) (*MCPMessage, error) {
	if msg.ID == nil {
		return nil, fmt.Errorf("request message must have an ID")
	}
	
	// Create response channel
	responseChan := make(chan *MCPMessage, 1)
	requestID := fmt.Sprintf("%v", msg.ID)
	
	c.requestMutex.Lock()
	c.pendingRequests[requestID] = responseChan
	c.requestMutex.Unlock()
	
	// Clean up on return
	defer func() {
		c.requestMutex.Lock()
		delete(c.pendingRequests, requestID)
		c.requestMutex.Unlock()
		close(responseChan)
	}()
	
	// Send message
	if err := c.sendMessage(msg); err != nil {
		return nil, err
	}
	
	// Wait for response
	ctx, cancel := context.WithTimeout(c.ctx, c.config.RequestTimeout)
	defer cancel()
	
	select {
	case response := <-responseChan:
		c.metrics.SuccessfulRequests++
		return response, nil
	case <-ctx.Done():
		c.metrics.FailedRequests++
		return nil, fmt.Errorf("request timeout")
	case <-c.ctx.Done():
		c.metrics.FailedRequests++
		return nil, fmt.Errorf("client disconnected")
	}
}

// handleMessage processes incoming messages
func (c *ProductionMCPClient) handleMessage(msg *MCPMessage) {
	// Handle responses to pending requests
	if msg.ID != nil && (msg.Result != nil || msg.Error != nil) {
		requestID := fmt.Sprintf("%v", msg.ID)
		c.requestMutex.RLock()
		if responseChan, exists := c.pendingRequests[requestID]; exists {
			select {
			case responseChan <- msg:
			default:
				c.logger.Printf("Warning: response channel full for request %s", requestID)
			}
		}
		c.requestMutex.RUnlock()
		return
	}
	
	// Handle notifications
	switch msg.Method {
	case "notifications/tools/list_changed":
		c.logger.Printf("Tools list changed")
		if c.eventHandlers.OnToolListChanged != nil {
			go c.eventHandlers.OnToolListChanged()
		}
		go c.loadTools()
		
	case "notifications/resources/list_changed":
		c.logger.Printf("Resources list changed")
		if c.eventHandlers.OnResourceListChanged != nil {
			go c.eventHandlers.OnResourceListChanged()
		}
		go c.loadResources()
		
	case "notifications/prompts/list_changed":
		c.logger.Printf("Prompts list changed")
		if c.eventHandlers.OnPromptListChanged != nil {
			go c.eventHandlers.OnPromptListChanged()
		}
		go c.loadPrompts()
		
	case "notifications/resources/updated":
		if params, ok := msg.Params.(map[string]interface{}); ok {
			c.handleResourceUpdateNotification(params)
		}
		
	case "notifications/message":
		if params, ok := msg.Params.(map[string]interface{}); ok {
			logEntryBytes, _ := json.Marshal(params)
			var logEntry MCPLogEntry
			if err := json.Unmarshal(logEntryBytes, &logEntry); err == nil {
				if c.eventHandlers.OnLogEntry != nil {
					go c.eventHandlers.OnLogEntry(&logEntry)
				}
			}
		}
		
	default:
		c.logger.Printf("Unknown notification method: %s", msg.Method)
	}
}

// handleResourceUpdateNotification handles resource update notifications
func (c *ProductionMCPClient) handleResourceUpdateNotification(params map[string]interface{}) {
	if c.eventHandlers.OnResourceUpdated == nil {
		return
	}
	
	uri, _ := params["uri"].(string)
	changeType, _ := params["type"].(string)
	
	// Fetch updated content in background
	go func() {
		if content, err := c.ReadResource(uri); err == nil {
			c.eventHandlers.OnResourceUpdated(uri, changeType, content)
		}
	}()
}

// handleDisconnection handles connection loss and implements reconnection logic
func (c *ProductionMCPClient) handleDisconnection(err error) {
	c.logger.Printf("Connection lost: %v", err)
	
	// Update state
	atomic.StoreInt32(&c.state, int32(MCPStateDisconnected))
	
	// Close connection
	c.connMutex.Lock()
	if c.conn != nil {
		c.conn.Close()
		c.conn = nil
	}
	c.connMutex.Unlock()
	
	// Trigger disconnected event
	if c.eventHandlers.OnDisconnected != nil {
		go c.eventHandlers.OnDisconnected(err)
	}
	
	// Attempt reconnection if enabled
	if c.reconnectAttempts < c.maxReconnects {
		c.reconnectAttempts++
		c.metrics.ReconnectCount++
		
		c.logger.Printf("Attempting reconnection %d/%d in %v", 
			c.reconnectAttempts, c.maxReconnects, c.reconnectDelay)
		
		time.Sleep(c.reconnectDelay)
		
		if err := c.Connect(); err != nil {
			c.logger.Printf("Reconnection failed: %v", err)
			// Exponential backoff
			c.reconnectDelay *= 2
			if c.reconnectDelay > 60*time.Second {
				c.reconnectDelay = 60 * time.Second
			}
		} else {
			c.logger.Printf("Successfully reconnected")
			c.reconnectAttempts = 0
			c.reconnectDelay = c.config.ReconnectDelay
		}
	} else {
		c.logger.Printf("Maximum reconnection attempts reached")
		if c.eventHandlers.OnError != nil {
			go c.eventHandlers.OnError(fmt.Errorf("maximum reconnection attempts reached"))
		}
	}
}

// SetEventHandlers sets the event handlers
func (c *ProductionMCPClient) SetEventHandlers(handlers MCPEventHandlers) {
	c.eventHandlers = handlers
}

// SetMetricsCallback sets a callback for metrics updates
func (c *ProductionMCPClient) SetMetricsCallback(callback func(*MCPClientMetrics)) {
	c.metricsCallback = callback
}

// GetMetrics returns current client metrics
func (c *ProductionMCPClient) GetMetrics() MCPClientMetrics {
	return *c.metrics
}

// Close gracefully closes the connection
func (c *ProductionMCPClient) Close() error {
	c.logger.Printf("Closing MCP client connection")
	
	// Cancel context
	if c.cancel != nil {
		c.cancel()
	}
	
	// Update state
	atomic.StoreInt32(&c.state, int32(MCPStateDisconnected))
	
	// Close WebSocket connection
	c.connMutex.Lock()
	if c.conn != nil {
		c.conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
		c.conn.Close()
		c.conn = nil
	}
	c.connMutex.Unlock()
	
	c.logger.Printf("MCP client connection closed")
	return nil
}

// GetState returns the current connection state
func (c *ProductionMCPClient) GetState() MCPConnectionState {
	return MCPConnectionState(atomic.LoadInt32(&c.state))
}

// IsConnected returns whether the client is connected
func (c *ProductionMCPClient) IsConnected() bool {
	return c.GetState() == MCPStateConnected
}

// GetServerInfo returns information about the connected server
func (c *ProductionMCPClient) GetServerInfo() *MCPServerInfo {
	c.dataMutex.RLock()
	defer c.dataMutex.RUnlock()
	return c.serverInfo
}

// GetTools returns available tools
func (c *ProductionMCPClient) GetTools() map[string]*MCPTool {
	c.dataMutex.RLock()
	defer c.dataMutex.RUnlock()
	
	tools := make(map[string]*MCPTool)
	for k, v := range c.tools {
		tools[k] = v
	}
	return tools
}

// GetResources returns available resources
func (c *ProductionMCPClient) GetResources() map[string]*MCPResource {
	c.dataMutex.RLock()
	defer c.dataMutex.RUnlock()
	
	resources := make(map[string]*MCPResource)
	for k, v := range c.resources {
		resources[k] = v
	}
	return resources
}

// GetPrompts returns available prompts
func (c *ProductionMCPClient) GetPrompts() map[string]*MCPPrompt {
	c.dataMutex.RLock()
	defer c.dataMutex.RUnlock()
	
	prompts := make(map[string]*MCPPrompt)
	for k, v := range c.prompts {
		prompts[k] = v
	}
	return prompts
}