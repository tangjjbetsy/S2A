import scipy.stats\
import argparse    

FEAT_EXTR = {
    'timbre': TimbreEvaluation,
    'midispec': MIDISpecEvaluation,
    'chroma': ChromaEvaluation,
}

def f_read_raw_mat(filename, col, data_format='f4', end='l'):
    f = open(filename,'rb')
    if end=='l':
        data_format = '<'+data_format
    elif end=='b':
        data_format = '>'+data_format
    else:
        data_format = '='+data_format
    datatype = np.dtype((data_format,(col,)))
    data = np.fromfile(f,dtype=datatype)
    f.close()
    if data.ndim == 2 and data.shape[1] == 1:
        return data[:,0]
    else:
        return data


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Objective Evaluation on MIDI-to-wav')
    parser.add_argument('--feat', 
                        type=str,
                        choices=['chroma', 'midispec'],
                        default='chroma', 
                        help='the type of feature to extract')
    parser.add_argument('con_sys',
                        type=str,
                        help='the directory of data information json file')
    parser.add_argument('exp_sys',
                        type=str,
                        help='the directory of data information json file')

    files = glob.glob("output/{}/{}/*.npy".format(args.feat, args.con_sys))
    feature_extractor_class = FEAT_EXTR[args.feat]
    model = feature_extractor_class()
    
    for i in files:
        exp_feat_dir = i
        con_feat_dir = i.replace(f"{args.con_sys}", f"{args.exp_sys}")
        g_dim = 512

        exp_feat = f_read_raw_mat(exp_feat_dir, col=g_dim)
        con_feat = f_read_raw_mat(con_feat_dir, col=g_dim)
        
        dis = model.compute_dis(con_feat, exp_feat)
        dis = np.round(dis, 4)
        dis_list.append(dis)

    m, h = mean_confidence_interval(dis_list)
    print("Mean: {}, Confidence Interval: +/- {}".format(m, h))