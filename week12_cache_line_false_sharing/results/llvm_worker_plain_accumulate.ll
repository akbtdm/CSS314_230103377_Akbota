define i32 @_ZN8__main__23worker_plain_accumulateB2v2B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedExxy(i8** noalias nocapture writeonly %retptr, { i8*, i32, i8*, i8*, i32 }** noalias nocapture readnone %excinfo, i8* nocapture readnone %arg.buf.0, i8* nocapture readnone %arg.buf.1, i64 %arg.buf.2, i64 %arg.buf.3, i64* nocapture %arg.buf.4, i64 %arg.buf.5.0, i64 %arg.buf.6.0, i64 %arg.slot, i64 %arg.iters, i64 %arg.seed) local_unnamed_addr #0 {
B0.endif:
  %.1063.not = icmp slt i64 %arg.iters, 1
  br i1 %.1063.not, label %common.ret, label %B30.lr.ph

B30.lr.ph:                                        ; preds = %B0.endif
  %.158 = icmp slt i64 %arg.slot, 0
  %.159 = select i1 %.158, i64 %arg.buf.5.0, i64 0
  %.160 = add i64 %.159, %arg.slot
  %.173 = getelementptr i64, i64* %arg.buf.4, i64 %.160
  %.173.promoted = load i64, i64* %.173, align 8
  %min.iters.check = icmp ult i64 %arg.iters, 4
  br i1 %min.iters.check, label %B30.preheader, label %vector.ph

vector.ph:                                        ; preds = %B30.lr.ph
  %n.vec = and i64 %arg.iters, -4
  %0 = insertelement <2 x i64> <i64 poison, i64 0>, i64 %.173.promoted, i64 0
  %broadcast.splatinsert = insertelement <2 x i64> poison, i64 %arg.seed, i64 0
  %broadcast.splat = shufflevector <2 x i64> %broadcast.splatinsert, <2 x i64> poison, <2 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %lsr.iv = phi i64 [ %lsr.iv.next, %vector.body ], [ %n.vec, %vector.ph ]
  %vec.phi = phi <2 x i64> [ %0, %vector.ph ], [ %3, %vector.body ]
  %vec.phi7 = phi <2 x i64> [ zeroinitializer, %vector.ph ], [ %4, %vector.body ]
  %vec.ind = phi <2 x i64> [ <i64 0, i64 1>, %vector.ph ], [ %vec.ind.next, %vector.body ]
  %step.add = add <2 x i64> %vec.ind, <i64 2, i64 2>
  %1 = xor <2 x i64> %vec.ind, %broadcast.splat
  %2 = xor <2 x i64> %step.add, %broadcast.splat
  %3 = add <2 x i64> %vec.phi, %1
  %4 = add <2 x i64> %vec.phi7, %2
  %vec.ind.next = add <2 x i64> %vec.ind, <i64 4, i64 4>
  %lsr.iv.next = add i64 %lsr.iv, -4
  %5 = icmp eq i64 %lsr.iv.next, 0
  br i1 %5, label %middle.block, label %vector.body, !llvm.loop !0

middle.block:                                     ; preds = %vector.body
  %bin.rdx = add <2 x i64> %4, %3
  %6 = call i64 @llvm.vector.reduce.add.v2i64(<2 x i64> %bin.rdx)
  %cmp.n = icmp eq i64 %n.vec, %arg.iters
  br i1 %cmp.n, label %B26.common.ret_crit_edge, label %B30.preheader

B30.preheader:                                    ; preds = %B30.lr.ph, %middle.block
  %.1776.ph = phi i64 [ %6, %middle.block ], [ %.173.promoted, %B30.lr.ph ]
  %.48.04.ph = phi i64 [ %n.vec, %middle.block ], [ 0, %B30.lr.ph ]
  br label %B30

B30:                                              ; preds = %B30.preheader, %B30
  %.1776 = phi i64 [ %.177, %B30 ], [ %.1776.ph, %B30.preheader ]
  %.48.04 = phi i64 [ %.119, %B30 ], [ %.48.04.ph, %B30.preheader ]
  %.119 = add nuw nsw i64 %.48.04, 1
  %.176 = xor i64 %.48.04, %arg.seed
  %.177 = add i64 %.1776, %.176
  %exitcond.not = icmp eq i64 %arg.iters, %.119
  br i1 %exitcond.not, label %B26.common.ret_crit_edge, label %B30, !llvm.loop !2

B26.common.ret_crit_edge:                         ; preds = %B30, %middle.block
  %.177.lcssa = phi i64 [ %6, %middle.block ], [ %.177, %B30 ]
  %sunkaddr = mul i64 %.160, 8
  %7 = bitcast i64* %arg.buf.4 to i8*
  %sunkaddr17 = getelementptr i8, i8* %7, i64 %sunkaddr
  %8 = bitcast i8* %sunkaddr17 to i64*
  store i64 %.177.lcssa, i64* %8, align 8
  br label %common.ret

common.ret:                                       ; preds = %B26.common.ret_crit_edge, %B0.endif
  store i8* null, i8** %retptr, align 8
  ret i32 0
}
