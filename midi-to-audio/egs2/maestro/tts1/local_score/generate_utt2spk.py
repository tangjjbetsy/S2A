import sys

"""
This script generate utt2spk from the input file (wav.scp/segments)
The speaker is "piano" on maestro dataset.
In the multi-instrument generation, the speaker could be different instruments.
"""

def main(src_file, utt2spk, utt2lang):
    f_src = open(src_file, 'r')
    lines_src = f_src.readlines()
    
    f_utt2spk = open(utt2spk + "_original", 'r')
    line_utt2spk = f_utt2spk.readlines()
    utt2spk_original = dict()
    
    for item in line_utt2spk:
        utt_id = item.strip().split()[0]
        spk_id = item.strip().split()[1]
        utt2spk_original[utt_id] = spk_id
        
    f_utt2lang = open(utt2lang + "_original", 'r')
    line_utt2lang = f_utt2lang.readlines()
    utt2lang_original = dict()    
    
    for item in line_utt2lang:
        utt_id = item.strip().split()[0]
        lang_id = item.strip().split()[1]
        utt2lang_original[utt_id] = lang_id

    f_utt2spk = open(utt2spk, 'w')
    f_utt2lang = open(utt2lang, 'w')

    for item in lines_src:
        utt_id = item.strip().split()[0]
        spk_id = utt2spk_original["_".join(utt_id.split("_")[:-1])]
        lang_id = utt2lang_original["_".join(utt_id.split("_")[:-1])]
        f_utt2spk.write("{} {}\n".format(utt_id, spk_id))
        f_utt2lang.write("{} {}\n".format(utt_id, lang_id))
    
    f_utt2spk.close()
    f_utt2lang.close()


if __name__ == '__main__':
    src = sys.argv[1]
    utt2spk = sys.argv[2]
    utt2lang = sys.argv[3]
    main(src, utt2spk, utt2lang)