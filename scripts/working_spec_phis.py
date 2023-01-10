"""
Simple script to see if all the specs in the list spec_names can be loaded.
"""
from grl.environment import load_spec

if __name__ == "__main__":
    spec_names = [
        'slippery_tmaze_5_two_thirds_up', 'example_7', 'tiger-alt', '4x3.95', 'cheese.95',
        'network', 'paint.95', 'shuttle.95', 'bridge-repair'
    ]
    bad_specs = []
    good_specs = []
    for spec_name in spec_names:
        try:
            spec = load_spec(spec_name)
            good_specs.append(spec_name)
        except NotImplementedError:
            bad_specs.append(spec_name)

    print(f"Specs not implemented yet: {bad_specs}")
    print(f"Good specs to run: {good_specs}")
