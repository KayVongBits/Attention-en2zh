# 导入 SentencePiece 库
import sentencepiece as spm
import os  # 新增：导入系统模块用于处理文件夹

def train(input_file, vocab_size, model_name, model_type, character_coverage, save_dir):
    """
    新增参数:
    - save_dir: 模型和词表文件的保存目录
    """
    # 1. 检查并创建保存目录（exist_ok=True 表示如果文件夹已存在则不报错）
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. 拼接完整的模型前缀路径，例如 "./saved_models/eng"
    # 这里直接用字符串拼接，在 VS Code 根目录逻辑下绝对好用
    full_model_prefix = f"{save_dir}/{model_name}"

    input_argument = (
        '--input=%s '
        '--model_prefix=%s '  # 接收完整的路径前缀
        '--vocab_size=%s '
        '--model_type=%s '
        '--character_coverage=%s '
        '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
    )

    # 3. 把 full_model_prefix 塞进命令里
    cmd = input_argument % (input_file, full_model_prefix, vocab_size, model_type, character_coverage)

    print(f"开始训练，文件将保存在: {full_model_prefix}.model 和 .vocab")
    spm.SentencePieceTrainer.Train(cmd)


def run():
    # ================= 新增：定义保存目录 =================
    # 在 VS Code 中，./ 代表项目根目录。这里指定存在项目根目录下的 saved_models 文件夹中
    output_dir = "./tokenizer"  # 你可以根据需要修改这个路径
    
    # ===== 英文分词器配置 =====
    # ⚠️ 友情提示：因为你在 VS Code 里运行，相对路径是根目录。
    # 你的语料库应该在项目根目录的 data 文件夹里，所以改成了 ./data/
    en_input = './data/corpus.en'      
    en_vocab_size = 32000               
    en_model_name = 'eng'               
    en_model_type = 'bpe'               
    en_character_coverage = 1.0         

    # 调用时增加 output_dir
    train(en_input, en_vocab_size, en_model_name, en_model_type, en_character_coverage, output_dir)

    # ===== 中文分词器配置 =====
    ch_input = './data/corpus.ch'      
    ch_vocab_size = 32000
    ch_model_name = 'chn'
    ch_model_type = 'bpe'
    ch_character_coverage = 0.9995      

    train(ch_input, ch_vocab_size, ch_model_name, ch_model_type, ch_character_coverage, output_dir)


def test():
    sp = spm.SentencePieceProcessor()
    text = "美国总统特朗普今日抵达夏威夷。"

    # 加载时也要加上你指定的目录路径
    sp.Load("./tokenizer/chn.model")

    print(sp.EncodeAsPieces(text))
    print(sp.EncodeAsIds(text))

    a = [12907, 277, 7419, 7318, 18384, 28724]
    print(sp.DecodeIds(a))


if __name__ == "__main__":
    run()
    # test()
