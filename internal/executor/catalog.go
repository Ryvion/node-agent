package executor

import (
	"encoding/json"
	"os"
)

type KindSpec struct {
	Image string   `json:"image"`
	Args  []string `json:"args"`
}

type Catalog struct {
	Kinds map[string]KindSpec `json:"kinds"`
}

var catalog *Catalog

func LoadCatalogFromEnv() {
	path := os.Getenv("AK_CATALOG_JSON")
	if path == "" {
		return
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return
	}
	var c Catalog
	if err := json.Unmarshal(b, &c); err != nil {
		return
	}
	catalog = &c
}

func LookupKind(kind string) (KindSpec, bool) {
	if catalog == nil || catalog.Kinds == nil {
		return KindSpec{}, false
	}
	ks, ok := catalog.Kinds[kind]
	return ks, ok
}
