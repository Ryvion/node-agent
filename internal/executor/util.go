package executor

import (
    "io"
    "net/http"
    "os"
)

func downloadToFile(url, path string) error {
    if url == "" { return nil }
    resp, err := http.Get(url)
    if err != nil { return err }
    defer resp.Body.Close()
    f, err := os.Create(path)
    if err != nil { return err }
    defer f.Close()
    _, err = io.Copy(f, resp.Body)
    return err
}
