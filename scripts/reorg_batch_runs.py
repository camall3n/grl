from pathlib import Path

from grl.utils.file_system import load_info, numpyify_and_save

from definitions import ROOT_DIR

if __name__ == "__main__":

    directory = Path(ROOT_DIR, 'results', "batch_run_pg")

    for results_path in directory.iterdir():
        if results_path.is_dir() or results_path.suffix != '.npy':
            continue

        info = load_info(results_path)

        amo = info['logs']['after_mem_op']
        if 'ld' in amo:
            print(f"Already parsed {results_path}")
            continue

        ld_dict = {'mems': amo['lambda_discrep_mems'], 'measures': amo['lambda_discrep_mems_measures'] }
        mstde_dict = { 'mems': amo['mstde_mems'], 'measures': amo['mstde_mems_measures'] }
        mstde_res_dict = { 'mems': amo['mstde_res_mems'], 'measures': amo['mstde_res_mems_measures'] }
        info['logs']['after_mem_op'] = {'ld': ld_dict, 'mstde': mstde_dict, 'mstde_res': mstde_res_dict}

        f = info['logs']['final']
        f_ld_dict = {'pi_params': f['ld_final_pi_params'], 'measures': f['lambda_discrep_final_measures']}
        f_mstde_dict = {'pi_params': f['mstde_final_pi_params'], 'measures': f['mstde_final_measures']}
        f_mstde_res_dict = {'pi_params': f['mstde_res_final_pi_params'], 'measures': f['mstde_res_final_measures']}
        info['logs']['final'] = {'ld': f_ld_dict, 'mstde': f_mstde_dict, 'mstde_res': f_mstde_res_dict}

        numpyify_and_save(results_path, info)

        print(f"Finished reorganizing results in {results_path}")


