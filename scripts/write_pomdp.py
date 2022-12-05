import numpy as np
from pathlib import Path
from typing import Union

from grl import load_spec
from definitions import ROOT_DIR

def convert_arr_to_lines(arr: np.ndarray) -> Union[str, list]:
    if len(arr.shape) == 1:
        return ' '.join(map(str, arr))
    elif len(arr.shape) == 2:
        return [convert_arr_to_lines(inner) for inner in arr]
    else:
        raise NotImplementedError

if __name__ == '__main__':
    # spec_name = 'tmaze_hyperparams'
    spec_name = 'slippery_tmaze_5_random'
    spec = load_spec(spec_name)
    pomdp_files_dir = Path(ROOT_DIR, 'grl', 'environment', 'pomdp_files')
    pomdp_path = pomdp_files_dir / 'slippery-tmaze.POMDP'

    lines = [f"# Converted POMDP file for {spec_name}"]
    lines.append('')

    lines.append(f"discount: {spec['gamma']}")
    lines.append(f"values: reward")
    lines.append(f"states: {spec['T'].shape[-1]}")
    lines.append(f"actions: {spec['T'].shape[0]}")
    lines.append(f"observations: {spec['phi'].shape[-1]}")
    lines.append('')

    lines.append('start:')
    lines.append(convert_arr_to_lines(spec['p0']))
    lines.append('')

    for i, T_a in enumerate(spec['T']):
        lines.append(f"T: {i}")
        lines += convert_arr_to_lines(T_a)
        lines.append('')
    lines.append('')

    lines.append("O: *")
    lines += convert_arr_to_lines(spec['phi'])
    lines.append('')

    for i, R_a in enumerate(spec['R']):
        for j, R_a_s in enumerate(R_a):
            for k, R_a_s_ns in enumerate(R_a_s):
                if abs(R_a_s_ns) > 0:
                    lines.append(f"R: {i} : {j} : {k} : * {R_a_s_ns}")
                    # lines += convert_arr_to_lines(R_a)
        lines.append('')
    lines.append('')

    with open(pomdp_path, 'w') as f:
        f.writelines([l + "\n" for l in lines])

    print(f"Written spec {spec_name} to POMDP file {pomdp_path}")