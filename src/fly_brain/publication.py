"""Portable provenance and anonymous ownership metadata for shared artifacts."""
import re
from pathlib import Path

_URL = re.compile(r'(https?://[^\s<>\"\'`]+)')
_HOME = r'/(?:Users|home)/[^/\s<>\"\'`]+'
_HOST_FIELDS = {'host', 'hostname', 'ssh_host', 'gpu_host', 'sensory_host'}


def portable(value, project_root=None, redactions=()):
    """Keep public URLs, normalize local paths, and redact configured private hosts."""
    if isinstance(value, dict):
        return {key: ('gpu-server' if key in _HOST_FIELDS and isinstance(item, str) and item not in ('', 'local')
                      else portable(item, project_root, redactions)) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item, project_root, redactions) for item in value]
    if not isinstance(value, str):
        return value
    # Host redactions also apply to private-host URLs; public account URLs stay intact.
    for name in sorted(redactions, key=len, reverse=True):
        if name:
            value = re.sub(re.escape(name), 'gpu-server', value, flags=re.IGNORECASE)
    parts = _URL.split(value)
    for index in range(0, len(parts), 2):
        text = parts[index]
        if project_root is not None:
            text = text.replace(str(Path(project_root).resolve()), 'project:')
        text = re.sub(_HOME + r'/code/codex/fly-brain(?:-codex)?(?=/|$)', 'project:', text)
        text = re.sub(_HOME + r'/code/data/malecns(?=/|$)', 'reference:', text)
        parts[index] = re.sub(_HOME, '~', text)
    return ''.join(parts)


def anonymous_tar_member(info):
    """Archives should not carry local account names, IDs, timestamps, or links."""
    if not info.isfile():
        raise ValueError('Publication archives accept regular files only')
    info.uid = info.gid = 0
    info.uname = info.gname = ''
    info.mtime = 0
    info.mode = 0o644
    info.pax_headers = {}
    return info
