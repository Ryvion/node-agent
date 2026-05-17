package sandbox

import "testing"

func TestEvaluateModelFormatDeclaredFormatWins(t *testing.T) {
	format := EvaluateModelFormat("model.gguf", "safetensors")
	if format != ModelFormatSafetensors {
		t.Fatalf("EvaluateModelFormat() = %q, want %q", format, ModelFormatSafetensors)
	}
}

func TestEvaluateModelFormatInfersKnownExtensions(t *testing.T) {
	tests := []struct {
		name       string
		pathOrName string
		want       ModelFormat
	}{
		{name: "gguf", pathOrName: "/models/llama.gguf", want: ModelFormatGGUF},
		{name: "safetensors", pathOrName: "model.safetensors", want: ModelFormatSafetensors},
		{name: "onnx", pathOrName: "encoder.onnx", want: ModelFormatONNX},
		{name: "torchscript", pathOrName: "ranker.torchscript", want: ModelFormatTorchScript},
		{name: "pytorch pt", pathOrName: "checkpoint.pt", want: ModelFormatPyTorchPickle},
		{name: "pytorch model bin", pathOrName: `C:\models\pytorch_model.bin`, want: ModelFormatPyTorchPickle},
		{name: "python source", pathOrName: "model.py", want: ModelFormatPythonSource},
		{name: "unknown", pathOrName: "model.weights", want: ModelFormatUnknown},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := EvaluateModelFormat(tt.pathOrName, ""); got != tt.want {
				t.Fatalf("EvaluateModelFormat() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestEvaluateModelFormatNormalizesUnsafeDeclarations(t *testing.T) {
	tests := []struct {
		declared string
		want     ModelFormat
	}{
		{declared: "PyTorch-Pickle", want: ModelFormatPyTorchPickle},
		{declared: ".pth", want: ModelFormatPyTorchPickle},
		{declared: "python source", want: ModelFormatPythonSource},
		{declared: "application/x-safetensors", want: ModelFormatSafetensors},
	}

	for _, tt := range tests {
		t.Run(tt.declared, func(t *testing.T) {
			if got := EvaluateModelFormat("", tt.declared); got != tt.want {
				t.Fatalf("EvaluateModelFormat() = %q, want %q", got, tt.want)
			}
		})
	}
}
