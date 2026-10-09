"""Connect audited OOD manifests to the existing graph/model pipeline."""

import hashlib
import json
from pathlib import Path

from utils.ood_splits import load_ood_manifest, manifest_digest, materialize_ood_posts


_MANIFESTS = {}


def _enabled(value):
    return str(value).strip().lower() in {'true', '1', 'yes', 'on'}


def prepare_ood(args):
    """Validate before encoder construction or any processed-cache shortcut."""
    if getattr(args, 'experiment_mode', 'id') != 'ood':
        raise ValueError('prepare_ood requires experiment_mode: ood')
    pattern = getattr(args, 'ood_manifest', None)
    if not pattern:
        raise ValueError('OOD requires ood_manifest; run scripts/prepare_ood_splits.py first.')
    path = Path(str(pattern).format(seed=args.seed)).expanduser().resolve()
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size, args.seed)
    if key not in _MANIFESTS:
        _MANIFESTS[key] = load_ood_manifest(path, seed=args.seed, validate_files=True)
    manifest = _MANIFESTS[key]
    if manifest['target_dataset'] != args.dataset:
        raise ValueError('Config dataset must equal manifest target_dataset.')
    if getattr(args, 'ood_val_domain', 'source') != 'source':
        raise ValueError('OOD model selection must use source validation only.')
    if getattr(args, 'early_test_root', None):
        raise ValueError('OOD uses the manifest test set; early_test_root would replace that protocol.')
    for option in ('p2t3_pretrained_path', 'kpg_checkpoint', 'checkpoint_path'):
        if getattr(args, option, None):
            raise ValueError('OOD source-only training cannot reuse {}. Remove this checkpoint.'.format(option))
    if _enabled(getattr(args, 'see_ttt_enabled', getattr(args, 'base_model', '') == 'SEEGraphMAE')):
        raise ValueError('Set see_ttt_enabled: false for source-only OOD evaluation.')
    if _enabled(getattr(args, 'kpg_test_only', False)):
        raise ValueError('OOD requires source training; kpg_test_only is unsupported.')
    embedding = getattr(args, 'word_embedding', '')
    datasets = set(manifest['source_roots'])
    if embedding == 'word2vec':
        if 'Pheme' in datasets and datasets & {'Weibo', 'DRWeibo'}:
            raise ValueError('Cross-language OOD requires a shared multilingual encoder; use multilingual-e5-base.')
        if getattr(args, 'word2vec_model_path', None):
            raise ValueError('OOD Word2Vec is trained on manifest train only; remove word2vec_model_path.')
        expected_language = 'en' if datasets == {'Pheme'} else 'ch'
        if args.language != expected_language:
            raise ValueError('OOD Word2Vec language must be {}.'.format(expected_language))
        args.in_feats = args.vector_size
    elif embedding == 'multilingual-e5-base':
        args.in_feats = getattr(args, '_ood_embedding_dim', 768)
    else:
        raise ValueError('Unsupported OOD word_embedding: {}'.format(embedding))
    args.ood_fingerprint = manifest_digest(manifest)
    args.ood_protocol = manifest['protocol']
    args.ood_resolved_manifest = str(path)
    args.ood_source_datasets = sorted({r['dataset'] for r in manifest['splits']['train']})
    args.ood_val_domain = 'source'
    # Existing EIN histories pad to this width. All three corpora must share it.
    args.max_hop = max(72, int(getattr(args, 'max_hop', 72)))
    return manifest


def ood_cache_token(args):
    prepare_ood(args)
    return args.ood_fingerprint[:20]


def ood_word2vec_path(args):
    token = ood_cache_token(args)
    settings = '{}-{}-{}'.format(args.language, args.tokenize_mode, args.vector_size)
    signature = hashlib.sha256(settings.encode()).hexdigest()[:12]
    return Path('word2vec') / 'ood' / '{}_{}.model'.format(token, signature)


def ood_training_sentences(args, tokenizer):
    manifest = prepare_ood(args)
    sentences = []
    for _, post in materialize_ood_posts(manifest, 'train'):
        for node in [post['source']] + post['comment']:
            sentences.append(tokenizer(node['content'], lang=args.language, mode=args.tokenize_mode))
    return sentences


def write_ood_provenance(args, directory):
    """Make cache origin reviewable without embedding the full manifest in logs."""
    manifest = prepare_ood(args)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'ood_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
