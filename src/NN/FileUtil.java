package NN;

import java.io.*;
import java.util.*;

/**
 * FileUtil allows the user to convert a text file to a String Iterator and convert a String Iterator
 * into a file.
 */
public class FileUtil
{
   /**
    * Returns a file in the form of a String Iterator.
    * @param fileName   The name of your file.
    * @return  The file in the form of a String Iterator.
    */
   public static Iterator<String> loadFile(String fileName)
   {
      try
      {
         Scanner in = new Scanner(new File(fileName));
         List<String> list = new ArrayList<String>();
         while (in.hasNextLine())
            list.add(in.nextLine());
         in.close();
         return list.iterator();
      }
      catch(FileNotFoundException e)
      {
         throw new RuntimeException(e);
      }
   }
   
   /**
    * Saves a text file given a filename and an Iterator of Strings.
    * @param fileName   The desired name of the file.
    * @param data       The text to be printed in the file.
    */
   public static void saveFile(String fileName, Iterator<String> data)
   {
      try
      {
         PrintWriter out = new PrintWriter(
                 new FileWriter(fileName), true);
         while (data.hasNext())
            out.println(data.next());
         out.close();
      }
      catch(IOException e)
      {
         throw new RuntimeException(e);
      }
   }
}
