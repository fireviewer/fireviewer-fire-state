from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
from typing import Protocol

class ObjectStorageError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ObjectMetadata:
    pathname: str
    size_bytes: int
    content_type: str | None


class ObjectStore(Protocol):
    def pathname_for(self, key: str) -> str: ...

    def uri_for(self, key: str) -> str: ...

    def uri_for_pathname(self, pathname: str) -> str: ...

    def finalize_tree(self, source_dir: Path, key: str) -> None: ...

    def delete_tree(self, key: str) -> None: ...

    def read_bytes(self, uri: str) -> bytes: ...

    def iter_bytes(self, uri: str, *, chunk_size: int) -> Iterator[bytes]: ...

    def materialize(self, uri: str, destination: Path) -> ObjectMetadata: ...

    def put_file(self, source_file: Path, uri: str) -> None: ...

    def head(self, uri: str) -> ObjectMetadata: ...

    def list_prefix(self, key: str, *, limit: int) -> list[ObjectMetadata]: ...


