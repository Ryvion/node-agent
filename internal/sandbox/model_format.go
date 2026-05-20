package sandbox

import "strings"

type ModelFormat string

const (
	ModelFormatGGUF          ModelFormat = "gguf"
	ModelFormatSafetensors   ModelFormat = "safetensors"
	ModelFormatONNX          ModelFormat = "onnx"
	ModelFormatTorchScript   ModelFormat = "torchscript"
	ModelFormatPyTorchPickle ModelFormat = "pytorch_pickle"
	ModelFormatPythonSource  ModelFormat = "python_source"
	ModelFormatUnknown       ModelFormat = "unknown"
)

func EvaluateModelFormat(pathOrName string, declaredFormat string) ModelFormat {
	if format := normalizeDeclaredModelFormat(declaredFormat); format != ModelFormatUnknown {
		return format
	}
	return inferModelFormatFromName(pathOrName)
}

func normalizeDeclaredModelFormat(declaredFormat string) ModelFormat {
	format := strings.ToLower(strings.TrimSpace(declaredFormat))
	format = strings.Trim(format, `"'`)
	format = strings.TrimPrefix(format, ".")
	format = strings.ReplaceAll(format, "-", "_")
	format = strings.ReplaceAll(format, " ", "_")

	switch {
	case format == "gguf" || strings.HasSuffix(format, "/gguf"):
		return ModelFormatGGUF
	case format == "safetensors" || format == "safe_tensors" || strings.Contains(format, "safetensors"):
		return ModelFormatSafetensors
	case format == "onnx" || strings.HasSuffix(format, "/onnx"):
		return ModelFormatONNX
	case format == "torchscript" || format == "torch_script" || format == "torch_jit":
		return ModelFormatTorchScript
	case format == "pytorch_pickle" || format == "pytorch" || format == "torch" ||
		format == "pickle" || format == "pt" || format == "pth" || strings.Contains(format, "pickle"):
		return ModelFormatPyTorchPickle
	case format == "python_source" || format == "python" || format == "py":
		return ModelFormatPythonSource
	default:
		return ModelFormatUnknown
	}
}

func inferModelFormatFromName(pathOrName string) ModelFormat {
	name := strings.ToLower(strings.TrimSpace(pathOrName))
	name = strings.Trim(name, `"'`)
	if name == "" {
		return ModelFormatUnknown
	}
	if cut := strings.IndexAny(name, "?#"); cut >= 0 {
		name = name[:cut]
	}

	switch {
	case strings.HasSuffix(name, ".gguf"):
		return ModelFormatGGUF
	case strings.HasSuffix(name, ".safetensors"):
		return ModelFormatSafetensors
	case strings.HasSuffix(name, ".onnx"):
		return ModelFormatONNX
	case strings.HasSuffix(name, ".torchscript") || strings.HasSuffix(name, ".ptl"):
		return ModelFormatTorchScript
	case strings.HasSuffix(name, ".pt") ||
		strings.HasSuffix(name, ".pth") ||
		strings.HasSuffix(name, ".ckpt") ||
		strings.HasSuffix(name, ".pkl") ||
		strings.HasSuffix(name, ".pickle") ||
		strings.HasSuffix(name, "/pytorch_model.bin") ||
		strings.HasSuffix(name, `\pytorch_model.bin`):
		return ModelFormatPyTorchPickle
	case strings.HasSuffix(name, ".py") || strings.HasSuffix(name, ".pyz"):
		return ModelFormatPythonSource
	default:
		return ModelFormatUnknown
	}
}
