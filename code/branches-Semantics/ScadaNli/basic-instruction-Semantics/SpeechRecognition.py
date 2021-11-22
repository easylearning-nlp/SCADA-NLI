from aip import AipSpeech
import os
from pyaudio import PyAudio,paInt16
import wave


APP_ID = '15868150'
API_KEY = 'VL06hAhD9su9PmE9u6lw1btN'
SECRET_KEY = 'XMUhlSP1CXlnWB4aaAlwFbQ6N9snCl0G'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)




def wav2pcm(wav_file):
    pcm_file = "%s.pcm" %(wav_file.split(".")[0])
    # 调用控制台实现wav转pcm
    os.system("ffmpeg -y  -i %s  -acodec pcm_s16le -f s16le -ac 1 -ar 16000 %s"%(wav_file,pcm_file))

    return pcm_file


def SpeechRecog(pcm_file):
    with open(pcm_file, 'rb') as fp:
        file_context = fp.read()
    context = client.asr(file_context, 'pcm', 16000, {
        'dev_pid': 1536,
    })
    # code = context.get("err_no")
    # if code == 0:
    res_str = context.get("result")
    # elif code == 3000 or 3301 or 3302 or 3310 or 3309 or 3311 or 3312 or 3308:
    #     res_str = "用户输入错误"
    # elif code == 3303 or 3307:
    #     res_str = "服务器连接失败"
    # else:
    #     res_str = "请求超限"
    return  res_str[0]

if __name__ == '__main__':
    pcm = wav2pcm("./cache/tmp.wav")
    # print(SpeechRecog("tmp.wav"))