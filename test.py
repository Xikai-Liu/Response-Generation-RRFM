##TODO## restore problem found
sess = tf.Session()
names=['wordemb_score','bleu1','bleu2','bleu3','sbleu1','sbleu2','sbleu3']
eval_score=np.zeros(shape=(7,))
new_saver = tf.train.import_meta_graph('ckpt/ex505050.ckpt-17100.meta')
new_saver.restore(sess,'ckpt/ex505050.ckpt-17100')
graph = tf.get_default_graph() 
#init_op = tf.global_variables_initializer()
with graph.as_default():
    session_conf = tf.ConfigProto(
    allow_soft_placement=FLAGS.allow_soft_placement,
    log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session()
with sess.as_default():
    new_saver = tf.train.import_meta_graph('ckpt/ex505050.ckpt-17100.meta')
    new_saver.restore(sess,'ckpt/ex505050.ckpt-17100.meta')
    for eval_batch in db.get_eval_batch(32):
        feed_dict={encoder_inputs:eval_batch[1],
               rencoder_inputs:eval_batch[3],
          keep_prob_placeholder:1.0,
               batch_size:len(eval_batch[1])
              }
        predict=sess.run([predicted_ids],feed_dict=feed_dict)
        result=predict
        result=list(np.transpose(result[0],[0,2,1]))
        for references,item in zip(eval_batch[2],result):
            best_hyp=item[0]
            best_hyp=[db.response_vocabs[i] for i in best_hyp if i>3]
            score=get_scores(references,best_hyp,word2vec_model.wv)
            eval_score+=score
    eval_score/=db.eval_cnt
    result_dict=dict(list(zip(names,eval_score)))  

sentence=[]
for eval_batch in db.get_eval_batch(32):
        feed_dict={encoder_inputs:eval_batch[1],
               rencoder_inputs:eval_batch[3],
              keep_prob_placeholder:1.0,
               batch_size:len(eval_batch[1])
              }
        predict=sess.run([predicted_ids],feed_dict=feed_dict)
        result=predict
        #print(result)
        #print(eval_batch[2])
        result=list(np.transpose(result[0],[0,2,1]))
        for references,item in zip(eval_batch[2],result):
            best_hyp=item[0]
            best_hyp=[db.response_vocabs[i] for i in best_hyp if i>3]    
            sentence.append(best_hyp)
a=get_distinct(sentence,2)
print(a)
