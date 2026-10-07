"""Write result CSVs atomically, tolerating brief Windows/OneDrive file locks."""
from pathlib import Path
import time
import uuid


def write_csv(frame, path):
    target = Path(path)
    temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
    frame.to_csv(temporary, index=False)
    for attempt in range(6):
        try:
            temporary.replace(target)
            return
        except OSError:
            if attempt == 5:
                raise
            time.sleep(.5 * (attempt + 1))
