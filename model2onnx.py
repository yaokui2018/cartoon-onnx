import os
import cv2
import numpy as np
import tensorflow._api.v2.compat.v1 as tf

tf.compat.v1.disable_eager_execution()
import network
import guided_filter


def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image

class Cartoonizer():
    def __init__(self, model_path="./saved_models"):
        self.model_path = model_path

    def load_model(self):
        with tf.Graph().as_default() as g:
            input_photo = tf.placeholder(tf.float32, [1, None, None, 3], name='input_photo')
            network_out = network.unet_generator(input_photo)
            final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3, name='final_out')

            all_vars = tf.trainable_variables()
            gene_vars = [var for var in all_vars if 'generator' in var.name]
            saver = tf.train.Saver(var_list=gene_vars)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)

            sess.run(tf.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint(self.model_path))

            self.sess = sess
            self.final_out = final_out
            self.input_photo = input_photo
            self.graph = g

    def inference(self, input_image):
        image = resize_crop(input_image)
        batch_image = image.astype(np.float32) / 127.5 - 1
        batch_image = np.expand_dims(batch_image, axis=0)
        output = self.sess.run(self.final_out, feed_dict={self.input_photo: batch_image})
        output = (np.squeeze(output) + 1) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        return output

    def export_onnx(self, output_path):
        # Freeze the graph
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess=self.sess,
            input_graph_def=self.graph.as_graph_def(),
            output_node_names=[self.final_out.op.name]
        )
        for node in constant_graph.node:
            print(node.name)
        # Save the frozen graph to a file
        with tf.io.gfile.GFile(output_path, "wb") as f:
            f.write(constant_graph.SerializeToString())

        # Convert the frozen graph to ONNX format
        import tf2onnx
        from onnx import shape_inference

        with tf.Graph().as_default() as g:
            tf.import_graph_def(constant_graph, name="")
            with tf.Session(graph=g) as sess:
                frozen_graph_def = sess.graph.as_graph_def()

        onnx_output_path = os.path.splitext(output_path)[0] + ".onnx"
        with tf.io.gfile.GFile(onnx_output_path, "wb") as f:
            onnx_graph = tf2onnx.convert.from_graph_def(frozen_graph_def, opset=11,
                                                        input_names=[self.input_photo.op.name + ":0"],
                                                        output_names=[self.final_out.op.name + ":0"])
            model_proto = onnx_graph[0]  # Extract the first element of the tuple
            model_proto = shape_inference.infer_shapes(model_proto)
            f.write(model_proto.SerializeToString())
        print("ONNX model is saved to", onnx_output_path)


if __name__ == '__main__':
    model = Cartoonizer()
    model.load_model()
    # Export the model to ONNX format: checkpoint -> .pb -> .onnx
    model.export_onnx("frozen_cartoonizer.pb")

    # 重点：为输入/输出节点添加名称、导出onnx时节点名需要加上:0
