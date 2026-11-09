"""DeepSeek adapter for the author's baseline state machine (no torch needed)."""
import argparse
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import urllib.request
sys.path.insert(0,str(Path(__file__).parent/'src'))
import utils
ROOT=Path(__file__).resolve().parent


def parse(text):
    match=re.search(r'<tool_call>(.*?)</tool_call>',text,re.S)
    if match:
        try:
            call=json.loads(match.group(1))
            if isinstance(call,dict) and isinstance(call.get('arguments'),dict) and isinstance(call.get('name'),str):
                return dict(type='tool',tool_name=call['name'],arguments=call['arguments'],raw_text=text)
        except (ValueError,TypeError):pass
    return dict(type='content',content=text,raw_text=text)


def run_episode(task,generate,rounds=6):
    st=utils.init_state(task,utils.SYSTEM_PROMPT_XML,record_mode='full',prompt_mode='current',require_reasoning=False,tool_format='xml')
    events=[]
    for _ in range(rounds):
        if st['done']:break
        out=parse(generate(st['messages'],st['tools']))
        trace=dict(parsed_output=out,model_raw_output=out['raw_text'])
        utils.step_state(st,out,trace_item=trace)
        events.append(dict(round=st['rounds'],parsed=out,tool_result=trace.get('tool_result'),done=st['done']))
    item=utils.finalize_state(st)
    item.update(events=events,parse_failures=sum('<tool_call>' in e['parsed']['raw_text'] and e['parsed']['type']!='tool' for e in events),
                policy_rejections=sum(str((e.get('tool_result') or {}).get('message','')).startswith('Tool call rejected:') for e in events),
                final_correct=bool(utils.item_final_eval(item)[-1]),strict_correct=bool(st['done'] and utils.item_final_eval(item)[-1]),done=st['done'],budget_exhausted=not st['done'],round_limit=rounds)
    return item


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mock',action='store_true',help='Local synthetic responses, no API')
    p.add_argument('--per-difficulty',type=int,default=1)
    p.add_argument('--runs',type=int,default=1)
    p.add_argument('--max-rounds',type=int,choices=[3,6],default=6)
    p.add_argument('--output',default='outputs/pilot.json')
    a=p.parse_args()
    if not 1<=a.per_difficulty<=50 or a.runs<1:p.error('Invalid sample size')
    env=ROOT/'.env'
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                k,v=line.split('=',1);os.environ.setdefault(k.strip(),v.strip())
    if not a.mock and not os.environ.get('DEEPSEEK_API_KEY'):
        p.error('Set DEEPSEEK_API_KEY locally in .env; never paste it into chat.')
    tasks=[];rng=random.Random(42)
    for difficulty in ('easy','medium','hard'):
        path=ROOT/f'data/task_sets/calculator_single_{difficulty}_test.json'
        tasks+=rng.sample(json.loads(path.read_text()),a.per_difficulty)
    usage=dict(calls=0,attempts=0,input_tokens=0,output_tokens=0,cached_input_tokens=0)
    api_records=[]
    max_calls=int(os.environ.get('PILOT_MAX_CALLS','540'))
    def api(messages,tools):
        # Keep official XML protocol; make the same tool schema available as HF chat template would.
        if usage['attempts'] >= max_calls: raise RuntimeError('API request cap reached')
        usage['attempts']+=1
        msgs=[dict(m) for m in messages]
        msgs[0]['content']+='\nAvailable tools:\n'+json.dumps(tools,ensure_ascii=False)
        body=dict(model='deepseek-flash',messages=msgs,temperature=1,max_tokens=1024,thinking=dict(type='disabled'))
        req=urllib.request.Request('https://api.deepseek.com/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+os.environ['DEEPSEEK_API_KEY'],'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=60) as response:d=json.load(response)
        usage['calls']+=1;usage['input_tokens']+=d.get('usage',{}).get('prompt_tokens',0);usage['output_tokens']+=d.get('usage',{}).get('completion_tokens',0)
        usage['cached_input_tokens']+=d.get('usage',{}).get('prompt_cache_hit_tokens',0)
        api_records.append(dict(response_id=d.get('id'),model=d.get('model'),usage=d.get('usage'),finish_reason=d['choices'][0].get('finish_reason')))
        return d['choices'][0]['message']['content'] or ''
    results=[];start=time.monotonic()
    out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise FileExistsError('Output already exists; use a new study directory')
    import platform
    def save(error=None):
        payload=dict(version='round-budget-1.0',mock=a.mock,config={**vars(a),'model':'deepseek-flash','temperature':1,'max_tokens':1024,'thinking':'disabled','max_rounds':a.max_rounds,'sample_seed':42,'python':platform.python_version()},usage=usage,api_records=api_records,seconds=time.monotonic()-start,results=results,error=error)
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2));tmp.replace(out)
    for run in range(a.runs):
        for task in tasks:
            if a.mock:
                # Exercise actual tool execution, then answer from tool output, not gold labels.
                expression=task['instruction'].split(':',1)[1].strip()
                count=[0]
                def generate(messages,tools):
                    count[0]+=1
                    if count[0]<=2:
                        return '<tool_call>'+json.dumps(dict(name='evaluate_expression',arguments=dict(expression=expression)))+'</tool_call>'
                    for msg in reversed(messages):
                        if '<tool_response>' in msg['content']:
                            result=json.loads(msg['content'].split('<tool_response>')[1].split('</tool_response>')[0])
                            return '\\boxed{'+str(result['result'])+'}'
                    raise RuntimeError('No tool response')
            else:generate=api
            task_start=time.monotonic()
            before=usage.copy()
            try:
                item=run_episode(task,generate,rounds=a.max_rounds)
            except Exception as exc:
                save(dict(type=type(exc).__name__,task_id=task['id'],run=run))
                raise RuntimeError('Run stopped; completed results saved. Inspect API settings and balance.') from None
            if a.max_rounds==6:
                replay=iter(e['parsed']['raw_text'] for e in item['events'])
                prefix=run_episode(task,lambda *_:next(replay),rounds=3)
                item['prefix3']=dict(final_correct=prefix['final_correct'],strict_correct=prefix['strict_correct'],done=prefix['done'],rounds=prefix['rounds'])
                item['late_rescue']=bool(item['strict_correct'] and not prefix['strict_correct'])
            item.update(run=run,seconds=time.monotonic()-task_start,api_usage={k:usage[k]-before[k] for k in usage});results.append(item)
            save()
            print(f"run {run+1}/{a.runs} task {len(results)} correct={item['final_correct']} mode={item['reasoning_mode']}",flush=True)
    print(json.dumps(dict(mock=a.mock,tasks=len(results),accuracy=sum(r['final_correct'] for r in results)/len(results),parse_failures=sum(r['parse_failures'] for r in results),usage=usage)))

if __name__=='__main__':main()
