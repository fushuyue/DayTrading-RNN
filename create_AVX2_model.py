with tf.Graph().as_default(), tf.device('/cpu:0'):
    
    with tf.name_scope('input'):
        x_ = tf.placeholder(tf.float32, [None, None, 36], name="input_x")
        y_ = tf.placeholder(tf.int32, [None, ], name="input_y")
        
        formal = tf.placeholder(tf.float32, [None, None, 36], name="formal_x")
        
        num_stock = tf.placeholder(tf.int32, name="number_of_stock")
        
        initializer = tf.contrib.layers.xavier_initializer()
        
        x_concat = tf.concat([formal,x_],axis=1)
        ones = tf.ones([num_stock,380,1])
        
        x_slice1 = x_concat[:,:,:22]
        x_slice2 = x_concat[:,:,23:]
        x_slice_final = tf.concat([x_slice1,ones,x_slice2],2)
        
        x_sum = tf.reduce_sum(formal,-1)
        x_indices = tf.equal(x_sum,0)

        size = tf.reduce_sum(tf.cast(x_indices,tf.int32),1)
        size = size[0]

        x_slice = tf.slice(x_slice_final,[0,size,1],[-1,-1,-1])
        x_return = tf.slice(x_concat,[0,20,0],[-1,-1,-1],name="x_return")

        
    with tf.name_scope('network'):
        
        conv = _create_network(x_slice)

        w1 = tf.Variable(initializer(shape=[20,64,5]) , name="W_final")
        conv_final = tf.nn.conv1d(conv, w1, stride=20, padding="VALID")
        logits = tf.reshape(conv_final,[-1,5], name="logit")
#         logits_temp = tf.reshape(conv_final[:,10:,:],[-1,5], name="logit_for_update")


    with tf.variable_scope("loss"):
        loss = tf.reduce_sum(tf.losses.sparse_softmax_cross_entropy(logits=logits,labels=y_))
        prob = tf.reduce_max(tf.nn.softmax(logits),1)
        pred = tf.argmax(tf.nn.softmax(logits),1)
        mysize = tf.cast((380-size)/20,tf.int32,name='prob_shape')
        
        prob11 = tf.reshape(prob,[num_stock,mysize])
        prob1 = tf.slice(prob11,[0,mysize-1],[-1,-1])
        prob1 = tf.reshape(prob1,[-1,],name='probabilitiy')
        
        pred1 = tf.reshape(pred,[num_stock,mysize])
        pred1 = tf.slice(pred1,[0,mysize-1],[-1,-1])
        pred1 = tf.reshape(pred1,[-1,],name='prediction')       
        
        
        correct_prediction = tf.equal(pred, tf.cast(y_, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
        values, indices = tf.nn.top_k(tf.nn.softmax(logits),2)
    
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),5.0)

    optimizer = tf.train.AdamOptimizer(0.0001)
    _train_op = optimizer.apply_gradients(zip(grads, tvars))  

    saver = tf.train.Saver(max_to_keep=100)
    
    config = tf.ConfigProto()  
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
#     config.gpu_options.per_process_gpu_memory_fraction = 0.4
    
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,model_dir + 'model.ckpt-100')
    
    
    name = model_dir + 'model' + '.ckpt'
    save_path = saver.save(sess,name,global_step=100)
    
    
    bs = 1
    x = pd.read_csv("/home/quant_it/share/x/fushuyue/300616.csv")
    x = x.values
    x = x.reshape(1,-1,20,36)
#     x[:,:,:,22] = 1

    feed_dict = {}  
    feed_dict[x_] = x[:,60:61,:,:].reshape(bs,-1,36)
    feed_dict[formal] = x[:,42:60,:,:].reshape(bs,-1,36)
#     feed_dict[formal] = np.zeros([bs,12*20,36])
    feed_dict[num_stock] = bs

    a,b = sess.run([pred1,prob1], feed_dict) 
    

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
