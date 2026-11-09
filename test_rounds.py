import unittest,json
from deepseek_rounds import run_episode,parse
T=dict(id=1,difficulty='easy',instruction='Compute 2+3',environments=[dict(name='CalculatorEnv',tools=['evaluate_expression'],parameters={})],expected=dict(answer='5'))
CALL='<tool_call>'+json.dumps(dict(name='evaluate_expression',arguments=dict(expression='2+3')))+'</tool_call>'
def run(seq,limit):
 it=iter(seq);return run_episode(T,lambda *_:next(it),rounds=limit)
class Tests(unittest.TestCase):
 def test_early_finish(self):
  for n in (3,6):
   x=run([CALL,r'\boxed{5}'],n);self.assertEqual(x['rounds'],2);self.assertTrue(x['strict_correct'])
 def test_three_cap(self):
  x=run(['continue']*3,3);self.assertEqual(x['rounds'],3);self.assertTrue(x['budget_exhausted']);self.assertFalse(x['strict_correct'])
 def test_late_recovery(self):
  seq=['continue']*3+[CALL,r'\boxed{5}']
  self.assertFalse(run(seq,3)['strict_correct']);self.assertTrue(run(seq,6)['strict_correct'])
 def test_success_on_last_round(self):
  x=run(['continue',CALL,r'\boxed{5}'],3);self.assertTrue(x['done']);self.assertFalse(x['budget_exhausted'])
 def test_policy_unchanged(self):
  seq=['Let me compute this carefully. '+CALL,CALL,r'\boxed{5}']
  for n in (3,6):self.assertEqual(run(seq,n)['policy_rejections'],1)
 def test_prefix_replay(self):
  x=run(['continue']*3+[r'\boxed{5}'],6)
  it=iter(e['parsed']['raw_text'] for e in x['events']);y=run_episode(T,lambda *_:next(it),3)
  self.assertEqual(y['rounds'],3);self.assertFalse(y['strict_correct'])
 def test_parse_invalid(self):
  for t in ('<tool_call>[]</tool_call>','<tool_call>null</tool_call>'):self.assertEqual(parse(t)['type'],'content')
 def test_wrong_answer(self):self.assertFalse(run([r'\boxed{6}'],3)['final_correct'])
if __name__=='__main__':unittest.main()
