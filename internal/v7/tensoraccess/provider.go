package tensoraccess

import (
	"context"

	"github.com/Ryvion/node-agent/internal/v7/tensorplane"
)

type TensorAccessProvider interface {
	Name() string
	Backend() string
	Capability(ctx context.Context) TensorAccessCapability
	ListLoadedModels(ctx context.Context) ([]LoadedTensorModel, error)
	GetPage(ctx context.Context, req TensorPageRequest) (tensorplane.TensorPage, error)
	GetQuery(ctx context.Context, req TensorQueryRequest) (tensorplane.AttentionQuery, error)
}
