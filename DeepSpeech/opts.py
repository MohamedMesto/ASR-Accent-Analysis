def add_decoder_args(parser):
    beam_args = parser.add_argument_group("Beam Decode Options",
                                          "Configurations options for the CTC Beam Search decoder")
    beam_args.add_argument('--top-paths', default=1, type=int, help='number of beams to return')
    beam_args.add_argument('--beam-width', default=10, type=int, help='Beam width to use')
    beam_args.add_argument('--lm-path', default=None, type=str,
                           help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
    beam_args.add_argument('--alpha', default=0.8, type=float, help='Language model weight')
    beam_args.add_argument('--beta', default=1, type=float, help='Language model word bonus (all words)')
    beam_args.add_argument('--cutoff-top-n', default=40, type=int,
                           help='Cutoff number in pruning, only top cutoff_top_n characters with highest probs in '
                                'vocabulary will be used in beam search, default 40.')
    beam_args.add_argument('--cutoff-prob', default=1.0, type=float,
                           help='Cutoff probability in pruning,default 1.0, no pruning.')
    beam_args.add_argument('--lm-workers', default=1, type=int, help='Number of LM processes to use')
    return parser

Notebook='Notebook'
print('MMM', 'able to read parser of test_attr.py')
if Notebook=='Notebook':
	main_path='/home/mmm2050/QU_DFKI_Thesis/Experimentation/ASR_Accent_Analysis_De'
elif Notebook=='colab':
	main_path='/content/ASR_Accent_Analysis_De'

def add_inference_args(parser):
    parser.add_argument('--cuda', action="store_true", help='Use cuda')
    parser.add_argument('--half', action="store_true",
                        help='Use half precision. This is recommended when using mixed-precision at training time')
    parser.add_argument('--decoder', default="greedy", choices=["greedy", "beam"], type=str, help="Decoder to use")
    parser.add_argument('--model_path', default=main_path+'/models/deepspeech_final.pth',
                        help='Path to model file created by training')
    return parser
