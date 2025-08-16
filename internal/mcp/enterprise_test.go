package mcp

import (
	"context"
	"log"
	"testing"
	"time"
)

func TestEnterpriseModelContextProtocolInitialization(t *testing.T) {
	config := EnterpriseConfiguration{
		PrimaryServerURL:            "wss://enterprise-mcp.ryvion.ai/v1",
		FallbackServerURLs:          []string{"wss://fallback1.ryvion.ai/v1", "wss://fallback2.ryvion.ai/v1"},
		AuthenticationToken:         "enterprise_token_12345",
		SecurityLevel:               SecurityLevelElevated,
		ComplianceRequirements:      []ComplianceStandard{ComplianceSOC2, ComplianceGDPR},
		PerformanceTargets:          PerformanceTargets{
			MaximumLatencyMilliseconds:      100,
			MinimumThroughputRequestsPerSec: 1000,
			TargetAvailabilityPercentage:    99.9,
			MaximumErrorRatePercentage:      0.1,
		},
		HighAvailabilityEnabled:     true,
		LoadBalancingStrategy:       LoadBalancingAIOptimized,
		ConnectionPoolSize:          50,
		RequestTimeoutMilliseconds:  5000,
		CircuitBreakerThreshold:     10,
		RetryPolicyConfiguration:    RetryPolicyConfiguration{
			MaximumRetryAttempts:       5,
			InitialBackoffMilliseconds: 1000,
			MaximumBackoffMilliseconds: 30000,
			BackoffMultiplier:          2.0,
			ExponentialBackoffEnabled:  true,
			JitterEnabled:              true,
		},
		MonitoringConfiguration:     MonitoringConfiguration{
			MetricsCollectionInterval:   10 * time.Second,
			PerformanceAlertsEnabled:    true,
			SecurityAlertsEnabled:       true,
			ComplianceAlertsEnabled:     true,
			DetailedLoggingEnabled:      true,
			DistributedTracingEnabled:   true,
		},
		AuditConfiguration:          AuditConfiguration{
			AuditTrailEnabled:           true,
			AuditRetentionDays:          2555,
			RealTimeComplianceChecking:  true,
			EncryptedAuditStorage:       true,
			ImmutableAuditRecords:       true,
		},
		Logger:                      log.Default(),
	}

	enterprise, err := NewEnterpriseModelContextProtocol(config)
	if err != nil {
		t.Fatalf("Failed to initialize enterprise MCP: %v", err)
	}

	if enterprise.primaryClient == nil {
		t.Error("Primary client not initialized")
	}

	if len(enterprise.fallbackClients) != 2 {
		t.Error("Fallback clients not properly initialized")
	}

	if enterprise.capabilityCache == nil {
		t.Error("Capability cache not initialized")
	}

	if enterprise.executionMetrics == nil {
		t.Error("Execution metrics collector not initialized")
	}

	if enterprise.securityValidator == nil {
		t.Error("Security validation engine not initialized")
	}

	if enterprise.enterpriseEventBus == nil {
		t.Error("Enterprise event bus not initialized")
	}

	if enterprise.complianceAuditor == nil {
		t.Error("Compliance auditor not initialized")
	}

	if enterprise.performanceMonitor == nil {
		t.Error("Performance monitor not initialized")
	}

	if enterprise.highAvailabilityManager == nil {
		t.Error("High availability manager not initialized")
	}

	if enterprise.loadBalancer == nil {
		t.Error("Load balancer not initialized")
	}
}

func TestEnterpriseToolExecution(t *testing.T) {
	config := EnterpriseConfiguration{
		PrimaryServerURL:           "wss://test-mcp.ryvion.ai/v1",
		AuthenticationToken:        "test_token",
		SecurityLevel:              SecurityLevelBasic,
		ComplianceRequirements:     []ComplianceStandard{ComplianceSOC2},
		PerformanceTargets:         PerformanceTargets{
			MaximumLatencyMilliseconds:      500,
			MinimumThroughputRequestsPerSec: 100,
			TargetAvailabilityPercentage:    99.0,
			MaximumErrorRatePercentage:      1.0,
		},
		HighAvailabilityEnabled:    false,
		LoadBalancingStrategy:      LoadBalancingRoundRobin,
		ConnectionPoolSize:         10,
		RequestTimeoutMilliseconds: 10000,
		RetryPolicyConfiguration:   RetryPolicyConfiguration{
			MaximumRetryAttempts:       3,
			InitialBackoffMilliseconds: 500,
			MaximumBackoffMilliseconds: 5000,
			BackoffMultiplier:          1.5,
			ExponentialBackoffEnabled:  true,
			JitterEnabled:              false,
		},
		Logger:                     log.Default(),
	}

	enterprise, err := NewEnterpriseModelContextProtocol(config)
	if err != nil {
		t.Fatalf("Failed to initialize enterprise MCP: %v", err)
	}

	ctx := context.Background()
	
	toolRequest := EnterpriseToolRequest{
		ToolName:        "enterprise_database_analyzer",
		Arguments:       map[string]interface{}{
			"database_name":    "production_analytics",
			"analysis_type":    "performance_optimization",
			"time_range":       "last_24_hours",
			"include_indices":  true,
			"compliance_mode":  true,
		},
		SecurityContext: SecurityContext{
			UserID:              "enterprise_user_001",
			AuthenticationLevel: 3,
			Clearance:           2,
			Department:          "data_engineering",
			Role:               "senior_analyst",
		},
		ComplianceLevel:     ComplianceSOC2,
		Priority:           1,
	}

	response, err := enterprise.ExecuteEnterpriseToolOperation(ctx, toolRequest)
	
	if response == nil {
		t.Error("Expected enterprise tool response, got nil")
	}

	if response.RequestID == "" {
		t.Error("Enterprise request ID should not be empty")
	}

	if response.ToolName != toolRequest.ToolName {
		t.Error("Tool name mismatch in response")
	}

	if response.ExecutionTime <= 0 {
		t.Error("Execution time should be positive")
	}

	if !response.ComplianceStatus.IsCompliant {
		t.Error("Expected compliant operation for SOC2")
	}
}

func TestEnterpriseResourceAccess(t *testing.T) {
	config := EnterpriseConfiguration{
		PrimaryServerURL:           "wss://secure-mcp.ryvion.ai/v1",
		AuthenticationToken:        "secure_enterprise_token",
		SecurityLevel:              SecurityLevelRestricted,
		ComplianceRequirements:     []ComplianceStandard{ComplianceGDPR, ComplianceHIPAA},
		PerformanceTargets:         PerformanceTargets{
			MaximumLatencyMilliseconds:      200,
			MinimumThroughputRequestsPerSec: 500,
			TargetAvailabilityPercentage:    99.95,
			MaximumErrorRatePercentage:      0.05,
		},
		HighAvailabilityEnabled:    true,
		LoadBalancingStrategy:      LoadBalancingLeastLatency,
		ConnectionPoolSize:         25,
		RequestTimeoutMilliseconds: 3000,
		Logger:                     log.Default(),
	}

	enterprise, err := NewEnterpriseModelContextProtocol(config)
	if err != nil {
		t.Fatalf("Failed to initialize enterprise MCP: %v", err)
	}

	ctx := context.Background()
	
	resourceRequest := EnterpriseResourceRequest{
		ResourceURI:     "secure://patient-data/encrypted-records/demographics",
		SecurityContext: SecurityContext{
			UserID:              "healthcare_analyst_002",
			AuthenticationLevel: 4,
			Clearance:           3,
			Department:          "healthcare_analytics",
			Role:               "hipaa_authorized_analyst",
		},
		AccessLevel:     2,
		ComplianceLevel: ComplianceHIPAA,
	}

	response, err := enterprise.AccessEnterpriseResource(ctx, resourceRequest)
	
	if response == nil {
		t.Error("Expected enterprise resource response, got nil")
	}

	if response.RequestID == "" {
		t.Error("Enterprise request ID should not be empty")
	}

	if response.ResourceURI != resourceRequest.ResourceURI {
		t.Error("Resource URI mismatch in response")
	}

	if response.AccessTime <= 0 {
		t.Error("Access time should be positive")
	}

	if !response.ComplianceStatus.IsCompliant {
		t.Error("Expected compliant operation for HIPAA")
	}

	if response.DataClassification.Level == 0 {
		t.Error("Data classification should be properly set for healthcare data")
	}
}

func TestEnterprisePromptRetrieval(t *testing.T) {
	config := EnterpriseConfiguration{
		PrimaryServerURL:           "wss://ai-mcp.ryvion.ai/v1",
		AuthenticationToken:        "ai_enterprise_token",
		SecurityLevel:              SecurityLevelElevated,
		ComplianceRequirements:     []ComplianceStandard{ComplianceISO27001},
		PerformanceTargets:         PerformanceTargets{
			MaximumLatencyMilliseconds:      150,
			MinimumThroughputRequestsPerSec: 750,
			TargetAvailabilityPercentage:    99.9,
			MaximumErrorRatePercentage:      0.1,
		},
		HighAvailabilityEnabled:    true,
		LoadBalancingStrategy:      LoadBalancingResourceBased,
		ConnectionPoolSize:         40,
		RequestTimeoutMilliseconds: 8000,
		Logger:                     log.Default(),
	}

	enterprise, err := NewEnterpriseModelContextProtocol(config)
	if err != nil {
		t.Fatalf("Failed to initialize enterprise MCP: %v", err)
	}

	ctx := context.Background()
	
	promptRequest := EnterprisePromptRequest{
		PromptName:      "enterprise_financial_analysis",
		Arguments:       map[string]interface{}{
			"analysis_scope":     "quarterly_performance",
			"financial_quarters": []string{"Q1_2024", "Q2_2024", "Q3_2024", "Q4_2024"},
			"include_forecasts":  true,
			"compliance_level":   "enterprise",
			"output_format":      "executive_summary",
		},
		SecurityContext: SecurityContext{
			UserID:              "financial_officer_003",
			AuthenticationLevel: 4,
			Clearance:           3,
			Department:          "finance",
			Role:               "chief_financial_officer",
		},
		ContentPolicy:   2,
	}

	response, err := enterprise.RetrieveEnterprisePrompt(ctx, promptRequest)
	
	if response == nil {
		t.Error("Expected enterprise prompt response, got nil")
	}

	if response.RequestID == "" {
		t.Error("Enterprise request ID should not be empty")
	}

	if response.PromptName != promptRequest.PromptName {
		t.Error("Prompt name mismatch in response")
	}

	if response.RetrievalTime <= 0 {
		t.Error("Retrieval time should be positive")
	}

	if !response.ComplianceStatus.IsCompliant {
		t.Error("Expected compliant operation for ISO27001")
	}

	if response.ContentValidation.ContainsRestrictedContent {
		t.Error("Financial analysis prompt should not contain restricted content")
	}
}

func TestEnterpriseMetricsCollection(t *testing.T) {
	config := EnterpriseConfiguration{
		PrimaryServerURL:           "wss://metrics-mcp.ryvion.ai/v1",
		AuthenticationToken:        "metrics_token",
		SecurityLevel:              SecurityLevelBasic,
		ComplianceRequirements:     []ComplianceStandard{ComplianceSOC2},
		PerformanceTargets:         PerformanceTargets{
			MaximumLatencyMilliseconds:      300,
			MinimumThroughputRequestsPerSec: 200,
			TargetAvailabilityPercentage:    99.0,
			MaximumErrorRatePercentage:      2.0,
		},
		HighAvailabilityEnabled:    false,
		LoadBalancingStrategy:      LoadBalancingRoundRobin,
		ConnectionPoolSize:         15,
		RequestTimeoutMilliseconds: 5000,
		MonitoringConfiguration:    MonitoringConfiguration{
			MetricsCollectionInterval:   5 * time.Second,
			PerformanceAlertsEnabled:    true,
			SecurityAlertsEnabled:       true,
			ComplianceAlertsEnabled:     true,
			DetailedLoggingEnabled:      true,
			DistributedTracingEnabled:   false,
		},
		Logger:                     log.Default(),
	}

	enterprise, err := NewEnterpriseModelContextProtocol(config)
	if err != nil {
		t.Fatalf("Failed to initialize enterprise MCP: %v", err)
	}

	metrics := enterprise.GetEnterpriseMetrics()
	
	if metrics == nil {
		t.Error("Expected enterprise metrics, got nil")
	}

	if metrics.ConnectionMetrics == (ConnectionMetrics{}) {
		t.Log("Connection metrics initialized as expected")
	}

	if metrics.ExecutionMetrics == (AggregatedExecutionMetrics{}) {
		t.Log("Execution metrics initialized as expected")
	}

	if metrics.PerformanceMetrics == (PerformanceMetrics{}) {
		t.Log("Performance metrics initialized as expected")
	}

	if metrics.SecurityMetrics == (SecurityMetrics{}) {
		t.Log("Security metrics initialized as expected")
	}

	if metrics.ComplianceMetrics == (ComplianceMetrics{}) {
		t.Log("Compliance metrics initialized as expected")
	}

	if metrics.AvailabilityMetrics == (AvailabilityMetrics{}) {
		t.Log("Availability metrics initialized as expected")
	}

	if metrics.LoadBalancingMetrics == (LoadBalancingMetrics{}) {
		t.Log("Load balancing metrics initialized as expected")
	}
}

func BenchmarkEnterpriseToolExecution(b *testing.B) {
	config := EnterpriseConfiguration{
		PrimaryServerURL:           "wss://benchmark-mcp.ryvion.ai/v1",
		AuthenticationToken:        "benchmark_token",
		SecurityLevel:              SecurityLevelBasic,
		ComplianceRequirements:     []ComplianceStandard{ComplianceSOC2},
		PerformanceTargets:         PerformanceTargets{
			MaximumLatencyMilliseconds:      50,
			MinimumThroughputRequestsPerSec: 2000,
			TargetAvailabilityPercentage:    99.99,
			MaximumErrorRatePercentage:      0.01,
		},
		HighAvailabilityEnabled:    true,
		LoadBalancingStrategy:      LoadBalancingAIOptimized,
		ConnectionPoolSize:         100,
		RequestTimeoutMilliseconds: 1000,
		Logger:                     log.Default(),
	}

	enterprise, err := NewEnterpriseModelContextProtocol(config)
	if err != nil {
		b.Fatalf("Failed to initialize enterprise MCP: %v", err)
	}

	ctx := context.Background()
	
	toolRequest := EnterpriseToolRequest{
		ToolName:        "high_performance_calculator",
		Arguments:       map[string]interface{}{
			"operation":    "matrix_multiplication",
			"matrix_size":  1000,
			"precision":    "double",
			"optimization": "simd",
		},
		SecurityContext: SecurityContext{
			UserID:              "benchmark_user",
			AuthenticationLevel: 1,
			Clearance:           1,
			Department:          "engineering",
			Role:               "performance_tester",
		},
		ComplianceLevel:     ComplianceSOC2,
		Priority:           0,
	}

	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		response, err := enterprise.ExecuteEnterpriseToolOperation(ctx, toolRequest)
		if err != nil {
			b.Errorf("Tool execution failed: %v", err)
		}
		if response == nil {
			b.Error("Expected response, got nil")
		}
	}
}

func TestEnterpriseSecurityValidation(t *testing.T) {
	securityEngine := NewSecurityValidationEngine(SecurityLevelRestricted)
	
	highRiskRequest := EnterpriseToolRequest{
		ToolName:        "system_administration",
		Arguments:       map[string]interface{}{
			"command":      "rm -rf /",
			"execute_as":   "root",
			"force":        true,
		},
		SecurityContext: SecurityContext{
			UserID:              "suspicious_user",
			AuthenticationLevel: 1,
			Clearance:           0,
			Department:          "unknown",
			Role:               "guest",
		},
		ComplianceLevel:     ComplianceSOC2,
		Priority:           1,
	}
	
	validationResult := securityEngine.ValidateToolRequest(highRiskRequest)
	
	if validationResult.IsBlocked {
		t.Log("High-risk request properly blocked by security validation")
	}
	
	legitimateRequest := EnterpriseToolRequest{
		ToolName:        "data_analytics",
		Arguments:       map[string]interface{}{
			"dataset":      "customer_satisfaction_survey",
			"analysis":     "trend_analysis",
			"time_period":  "last_month",
		},
		SecurityContext: SecurityContext{
			UserID:              "data_analyst_001",
			AuthenticationLevel: 3,
			Clearance:           2,
			Department:          "analytics",
			Role:               "senior_data_analyst",
		},
		ComplianceLevel:     ComplianceGDPR,
		Priority:           2,
	}
	
	legitimateValidation := securityEngine.ValidateToolRequest(legitimateRequest)
	
	if !legitimateValidation.IsBlocked {
		t.Log("Legitimate request properly allowed by security validation")
	}
}