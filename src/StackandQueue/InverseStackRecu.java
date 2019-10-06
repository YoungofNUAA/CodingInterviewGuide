package StackandQueue;

import java.util.Stack;

/**
 * ���õݹ麯����ջ��������һ��ջ
 * @author Young
 *
 */
public class InverseStackRecu {

	/**
	 * 
	 * @param stack
	 * @return  ���ز��Ƴ�ջ��Ԫ��
	 */
	public int getAndRemoveLastElement(Stack<Integer> stack) {
		int result = stack.pop();
		if(stack.isEmpty()) {
			return result;
		}else {
			int last = getAndRemoveLastElement(stack);
			stack.push(result);
			return last;
		}
	}
	
	public void reverse(Stack<Integer>stack) {
		if(stack.isEmpty()) {
			return;
		}
		int i = getAndRemoveLastElement(stack);
		reverse(stack);
		stack.push(i);
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		//����
//		Stack<Integer> stack = new Stack<Integer>();
//		stack.push(1);
//		stack.push(2);
//		stack.push(3);
//		InverseStackRecu test = new InverseStackRecu();
//		test.getAndRemoveLastElement(stack);
	}

}
