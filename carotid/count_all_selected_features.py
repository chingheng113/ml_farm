from carotid import carotid_data_util as cdu
import csv

if __name__ == '__main__':
    targets = ['RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA', 'LCCA', 'LEICA', 'LIICA', 'LACA',
               'LMCA', 'LPCA', 'LEVA', 'LIVA']
    source = 'exin' # 'ex'
    with open(source+'.csv', 'w', newline="") as csv_file:
        for target in targets:
            if source == 'ex':
                id_all, x_data_all, y_data_all = cdu.get_ex_fs_data(target)
            else:
                id_all, x_data_all, y_data_all = cdu.get_exin_fs_data(target)
            wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
            wr.writerow([target])
            wr.writerow(x_data_all.columns.values)
            print(x_data_all.shape[1])
