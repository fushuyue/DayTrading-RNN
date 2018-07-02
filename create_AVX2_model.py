with tf.Graph().as_default():
    config = tf.ConfigProto()  
    config.allow_soft_placement = True
    sess = tf.Session(config = config)
    saver = tf.train.import_meta_graph(model_dir + 'model.ckpt-100.meta')
    saver.restore(sess,model_dir + 'model.ckpt-100')
    out = [n.name for n in tf.get_default_graph().as_graph_def().node]
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
    sess,sess.graph_def,['input/number_of_stock','input/x_return','input/formal_x',
                         'loss/prediction','loss/probabilitiy','input/x_return','loss/prob_shape'] )
                         
    with open('output_graph.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())
