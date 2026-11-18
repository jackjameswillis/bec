"""Default output-stem builders shared by the experiment runners."""


def es_stem(args):
    return (
        f"es_{args.fitness}_{args.loss}_h{args.hidden}x{args.layers}"
        f"_{args.scale}_{args.activation}_p{args.population}_lr{args.lr}"
        f"_s{args.sigma}"
        f"{'_bl' + str(args.bias_lr) if args.bias_lr is not None else ''}"
        f"_seed{args.seed}_it{args.iterations}"
        f"_t{args.table_rows}{'_anti' if args.antithetic else ''}"
        f"{'_sh' + args.shaping if args.shaping != 'none' else ''}"
        f"{'_adam' if args.adam else ''}"
        f"{'_nat' if args.natural else ''}"
        f"{'_diag' if args.diagnostics else ''}"
    )


def ges_stem(args):
    return (
        f"ges_{args.fitness}_{args.loss}_h{args.hidden}x{args.layers}"
        f"_{args.scale}_{args.activation}_p{args.population}_lr{args.lr}"
        f"_s{args.sigma}_is{args.init_std}"
        f"_seed{args.seed}_it{args.iterations}"
        f"_t{args.table_rows}{'_anti' if args.antithetic else ''}"
        f"{'_sh' + args.shaping if args.shaping != 'none' else ''}"
        f"{'_adam' if args.adam else ''}"
        f"{'_diag' if args.diagnostics else ''}"
    )


def ste_stem(args, timestamp):
    return (
        f"ste_h{args.hidden}{'x' + str(args.layers) if args.layers != 1 else ''}"
        f"{'_' + args.scale if args.scale != 'none' else ''}"
        f"{'_' + args.optimizer if args.optimizer != 'sgd' else ''}"
        f"{'_noiq' if args.no_input_quant else ''}"
        f"{'_bias' if args.use_bias else ''}"
        f"_lr{args.learning_rate}_e{args.epochs}_b{args.batch_size}"
        f"_seed{args.seed}_{timestamp}"
    )
