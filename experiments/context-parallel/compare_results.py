"""Compare recorded tiny-runner outputs without loading a model or device."""
import argparse
import json
from pathlib import Path


def compare(records):
    baseline_name = 'TP2_PCP1_DCP1'
    baseline = records[baseline_name]
    expected = baseline['results'][0]['outputs']
    comparisons = []
    for name, record in sorted(records.items()):
        assert record['checkpoint_sha256'] == baseline['checkpoint_sha256']
        assert record['model_config'] == baseline['model_config']
        repetition_rows = []
        first = record['results'][0]['outputs']
        for rep in record['results']:
            actual = rep['outputs']
            assert [r['prompt_length'] for r in actual] == [r['prompt_length'] for r in expected]
            for row in actual:
                assert len(row['token_ids']) == len(row['chosen_logprobs']) == 4
            same_tokens = [r['token_ids'] for r in actual] == [r['token_ids'] for r in expected]
            differences = ([abs(a - b) for row, ref in zip(actual, expected)
                            for a, b in zip(row['chosen_logprobs'], ref['chosen_logprobs'])]
                           if same_tokens else None)
            repetition_rows.append({
                'repetition': rep['repetition'],
                'generated_token_ids_match_baseline': same_tokens,
                'chosen_logprob_max_abs_difference': max(differences) if differences is not None else None,
                'chosen_logprob_bitwise_equal': all(d == 0 for d in differences) if differences is not None else None,
                'reason_if_logprob_comparison_omitted': None if same_tokens else 'Generated token histories differ',
            })
        same_within = all([r['token_ids'] for r in rep['outputs']] == [r['token_ids'] for r in first]
                          for rep in record['results'])
        within_differences = ([abs(a - b) for rep in record['results'] for row, ref in zip(rep['outputs'], first)
                               for a, b in zip(row['chosen_logprobs'], ref['chosen_logprobs'])]
                              if same_within else None)
        comparisons.append({'configuration': name, 'topology': record['topology'],
                            'local_kv_heads': record['local_kv_heads'],
                            'cache_configuration': record['cache_configuration'],
                            'within_configuration': {
                                'generated_token_ids_match_first_repetition': same_within,
                                'chosen_logprob_max_abs_difference_from_first_repetition': (
                                    max(within_differences) if within_differences is not None else None),
                            },
                            'repetitions': repetition_rows})
    return {
        'schema': 'wiki-cp-tiny-runner-comparison/1',
        'baseline': baseline_name,
        'checkpoint_sha256': baseline['checkpoint_sha256'],
        'prompts': [row['prompt_length'] for row in expected],
        'generation': 'Greedy, 4 output tokens per prompt, 2 recorded repetitions per configuration',
        'all_recorded_tokens_match_baseline': all(r['generated_token_ids_match_baseline']
                                                 for c in comparisons for r in c['repetitions']),
        'comparison': comparisons,
        'limits': [
            'Matching greedy token IDs does not establish bitwise numerical equality.',
            'Chosen-token logprobs compare only the reported token positions, not full vocabulary logits.',
            'Random tiny-model execution does not assess pretrained model quality.',
            'Recorded wall times include profiler effects and are not a throughput or latency benchmark.',
            'Cache capacity fields are configured logical capacity, not measured maximum concurrent requests.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = {}
    for path in sorted(args.results.glob('tiny-TP*_PCP*_DCP*.json')):
        record = json.loads(path.read_text())
        records[path.stem.removeprefix('tiny-')] = record
    result = compare(records)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'configurations': len(records),
                      'all_recorded_tokens_match_baseline': result['all_recorded_tokens_match_baseline']}))


if __name__ == '__main__':
    main()
