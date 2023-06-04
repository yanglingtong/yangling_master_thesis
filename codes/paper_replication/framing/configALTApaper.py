args = {
    'model_type': 'roberta',
    'model_name': 'roberta-base', #'xlnet-base-cased',
    'task_name': 'multiclass',
    'max_seq_length': 256,
    'output_mode': 'classification',
    'train_batch_size': 8,
    'eval_batch_size': 8, # 32

    'gradient_accumulation_steps': 1,
    'num_train_epochs': 10, # 15
    'weight_decay': 5e-7,
    'learning_rate': 4e-5,
    'adam_epsilon': 1e-8,
    'warmup_steps': 10,
    'max_grad_norm': 1.0,
    'early_stop': 50,


    'notes': 'MFC dataset'
}


t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args['weight_decay']},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args['warmup_steps'], t_total=t_total)




for _ in train_iterator:
    for step, batch in enumerate(tqdm(train_dataloader)):
        model.train()
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,
                  'labels':         batch[3]}
        outputs = model(**inputs)
        loss = outputs[0]

        if args['gradient_accumulation_steps'] > 1:
            loss = loss / args['gradient_accumulation_steps']

        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

        tr_loss += loss.item()
        if (step + 1) % args['gradient_accumulation_steps'] == 0:
            scheduler.step()
            optimizer.step()
            model.zero_grad()
            global_step += 1

            if args['logging_steps'] > 0 and global_step % args['logging_steps'] == 0:
                # Log metrics
                if args['evaluate_during_training']:
                    results = evaluate(model, tokenizer)
                    for key, value in results.items():
                        tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args['logging_steps'], global_step)
                logging_loss = tr_loss

            if args['save_steps'] > 0 and global_step % args['save_steps'] == 0:
                # Save model checkpoint
                output_dir = os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model
                logger.info("Saving model checkpoint to %s", output_dir)
